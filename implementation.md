# Vision_Assist_2 — Raspberry Pi 5 Implementation Guide

> Complete guide to deploy the wearable assistant system on Raspberry Pi 5 for real-time, portable use by visually impaired users.

---

## Table of Contents
1. [Hardware Requirements](#1-hardware-requirements)
2. [OS & Environment Setup](#2-os--environment-setup)
3. [TTS Engine — Cross-Platform Fix](#3-tts-engine--cross-platform-fix)
4. [YOLO Optimization — NCNN Backend](#4-yolo-optimization--ncnn-backend)
5. [Ollama Edge Configuration](#5-ollama-edge-configuration)
6. [Performance Tuning — RuntimeConfig](#6-performance-tuning--runtimeconfig)
7. [Voice Commands on Pi](#7-voice-commands-on-pi)
8. [Camera Setup](#8-camera-setup)
9. [Power & Thermal Management](#9-power--thermal-management)
10. [Auto-Start on Boot](#10-auto-start-on-boot)
11. [Deployment Checklist](#11-deployment-checklist)

---

## 1. Hardware Requirements

### Minimum Setup (~₹8,000 / $95)
| Component | Model | Why |
|-----------|-------|-----|
| **Board** | Raspberry Pi 5 (8GB RAM) | 8GB is mandatory — YOLO + Ollama together use ~3-4GB |
| **Camera** | Pi Camera Module 3 or USB webcam | Pi Camera 3 has better autofocus for outdoor use |
| **Microphone** | USB mini mic or I2S MEMS mic (e.g., INMP441) | For voice commands |
| **Speaker** | 3.5mm mini speaker or Bluetooth earpiece | For TTS audio output |
| **Storage** | 32GB+ microSD (A2 class) | A2 class is critical for fast model loading |
| **Power** | 5V 5A USB-C PD power supply | Pi 5 draws up to 25W under load |

### Recommended Upgrades
| Component | Model | Benefit |
|-----------|-------|---------|
| **AI Accelerator** | Google Coral USB Accelerator (~$30) | YOLO jumps from 2 FPS to 15-25 FPS |
| **Storage** | NVMe SSD via Pi 5 HAT | 10x faster model loading vs microSD |
| **Cooling** | Active cooler / heatsink fan | Prevents thermal throttling under sustained AI load |
| **Battery** | 20,000mAh PD power bank | ~3-4 hours portable runtime |
| **Earpiece** | Bone conduction headphones | Keeps ears open for environment awareness |

---

## 2. OS & Environment Setup

### Step 1: Flash Raspberry Pi OS (64-bit)
```bash
# Use Raspberry Pi Imager to flash:
# Raspberry Pi OS (64-bit, Bookworm) — Desktop or Lite
# Enable SSH and Wi-Fi in imager settings
```

### Step 2: System Dependencies
```bash
sudo apt update && sudo apt upgrade -y

# Core dependencies
sudo apt install -y python3-pip python3-venv python3-dev \
    libopencv-dev portaudio19-dev libespeak-ng1 \
    ffmpeg git cmake build-essential

# For Pi Camera
sudo apt install -y libcamera-dev python3-picamera2
```

### Step 3: Python Environment
```bash
cd ~/Vision_Assist_2
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Ollama on Pi
```bash
curl -fsSL https://ollama.com/install.sh | sh

# Pull the lightweight model
ollama pull qwen2.5:0.5b

# Or the slightly smarter alternative
ollama pull llama3.2:1b
```

---

## 3. TTS Engine — Cross-Platform Fix

**CRITICAL:** The current TTS uses `win32com.client` (Windows SAPI5). This will NOT work on Linux/Pi. You must replace it.

### Option A: Piper TTS (Recommended — Fast, Offline, High Quality)

Piper is a neural TTS engine specifically optimized for Raspberry Pi. It runs entirely offline with ~50ms latency.

**Install Piper:**
```bash
pip install piper-tts

# Download a voice model (Amy = clear, natural English)
mkdir -p ~/piper-voices
cd ~/piper-voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

**Replace `_tts_process_worker` in `audio/tts.py`:**
```python
import platform
import subprocess
import os

def _tts_process_worker(ipc_queue, rate, volume):
    system = platform.system()

    if system == "Windows":
        # Windows: Use SAPI5 via win32com
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Volume = int(volume * 100)
        sapi_rate = max(-10, min(10, int((rate - 200) / 10)))
        speaker.Rate = sapi_rate

        while True:
            text = ipc_queue.get()
            if text is None:
                break
            try:
                _tts_speaking_event.set()
                speaker.Speak(text)
            except Exception:
                pass
            finally:
                _tts_speaking_event.clear()
    else:
        # Linux/Pi: Use Piper TTS
        voice_path = os.path.expanduser("~/piper-voices/en_US-amy-medium.onnx")

        while True:
            text = ipc_queue.get()
            if text is None:
                break
            try:
                _tts_speaking_event.set()
                # Pipe text through piper to aplay
                proc = subprocess.Popen(
                    ["piper", "--model", voice_path, "--output-raw"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                )
                audio_data, _ = proc.communicate(input=text.encode("utf-8"))
                # Play raw audio
                play = subprocess.Popen(
                    ["aplay", "-r", "22050", "-f", "S16_LE", "-c", "1", "-"],
                    stdin=subprocess.PIPE,
                )
                play.communicate(input=audio_data)
            except Exception as exc:
                print(f"TTS error: {exc}")
            finally:
                _tts_speaking_event.clear()
```

### Option B: espeak-ng (Simpler, Lower Quality)
```bash
sudo apt install espeak-ng

# Test it
espeak-ng "Hello, I am your assistant"
```

```python
# In _tts_process_worker for Linux:
import subprocess
subprocess.run(["espeak-ng", "-s", str(rate), text])
```

| Feature | Piper TTS | espeak-ng |
|---------|-----------|-----------|
| Voice Quality | Natural | Robotic |
| Latency | ~50ms | ~20ms |
| Offline | Yes | Yes |
| Model Size | ~60MB | ~2MB |
| Pi 5 CPU Usage | ~5% | ~1% |

---

## 4. YOLO Optimization — NCNN Backend

YOLOv8n on Pi 5 raw CPU runs at ~2 FPS. With NCNN export, you can reach 5-8 FPS.

### Step 1: Export YOLO to NCNN Format
```python
# Run this ONCE on your PC to export the model
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="ncnn", imgsz=320)
# This creates: yolov8n_ncnn_model/
```

### Step 2: Copy to Pi
```bash
scp -r yolov8n_ncnn_model/ pi@raspberrypi:~/Vision_Assist_2/
```

### Step 3: Update `vision/vision.py`
```python
@dataclass
class DetectorConfig:
    model_name: str = "yolov8n_ncnn_model"  # Changed from yolov8n.pt
    image_size: int = 320                    # Reduced from 416
    confidence_threshold: float = 0.45       # Slightly lower to catch more
    # ... rest stays the same
```

### Performance Comparison on Pi 5
| Config | FPS | Latency | RAM |
|--------|-----|---------|-----|
| YOLOv8n.pt, 416px | ~2 FPS | 500ms | ~800MB |
| YOLOv8n NCNN, 320px | ~6 FPS | 170ms | ~400MB |
| YOLOv8n NCNN, 256px | ~10 FPS | 100ms | ~300MB |
| YOLOv8n + Coral USB, 320px | ~20 FPS | 50ms | ~200MB |

### Optional: Coral USB Accelerator
```bash
# Install Coral runtime
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
    sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update && sudo apt install libedgetpu1-std

# Export model for Edge TPU (run on PC)
model.export(format="edgetpu", imgsz=320)
```

---

## 5. Ollama Edge Configuration

### Model Selection for Pi 5

| Model | Size | RAM | Speed (Pi 5) | Quality |
|-------|------|-----|--------------|---------|
| `qwen2.5:0.5b` | 350MB | ~600MB | 15-25 tok/s | Good |
| `llama3.2:1b` | 1.3GB | ~1.5GB | 5-10 tok/s | Better |
| `phi3` (original) | 2.3GB | ~3GB | 2-4 tok/s | Best but too slow |

### Recommended: `qwen2.5:0.5b` for Pi 5
```bash
ollama pull qwen2.5:0.5b
```

Update `llm/ollama_local.py`:
```python
DEFAULT_MODEL = "qwen2.5:0.5b"
DEFAULT_TIMEOUT_SECONDS = 8
```

### Ollama Performance Tuning
```bash
# Keep model loaded in RAM permanently (prevents cold-start delay)
# This is already done in code via keep_alive: -1

# Set Ollama to use all 4 cores
export OLLAMA_NUM_THREADS=4

# Limit context window for faster generation
export OLLAMA_NUM_CTX=512
```

### Alternative: Cloud-Only Mode (No Local LLM)
If RAM is too tight, you can skip Ollama entirely and rely on Gemini cloud for all reasoning. Modify `_async_llm_worker` in `main.py` to always call cloud instead of local.

---

## 6. Performance Tuning — RuntimeConfig

Update `main.py` at the bottom for Pi-optimized settings:

```python
if __name__ == "__main__":
    system = RealTimeAISystem(RuntimeConfig(
        frame_stride=5,                      # Process 1 in 5 frames (was 3)
        queue_size=1,                        # Minimal memory (was 2)
        min_recompute_interval_sec=3.0,      # Less frequent YOLO runs (was 2.0)
        inference_min_interval_sec=1.0,      # Slower loop cadence (was 0.35)
        cloud_min_interval_sec=6.0,          # Less frequent cloud calls (was 4.0)
        speech_keepalive_interval_sec=15.0,  # Less frequent keepalive (was 12.0)
        mode_status_interval_sec=12.0,       # Less frequent status (was 8.0)
    ))
    system.start()
```

### Expected Performance on Pi 5 (Optimized)

| Metric | PC (Current) | Pi 5 (Optimized) |
|--------|-------------|-------------------|
| YOLO inference | ~50ms | ~170ms (NCNN 320px) |
| Local LLM response | ~2s | ~3-4s (qwen2.5:0.5b) |
| Cloud API response | ~1-2s | ~1-2s (same) |
| Obstacle warning latency | ~100ms | ~300ms |
| End-to-end loop | ~0.5s | ~1.5s |
| RAM usage | ~2GB | ~2.5GB |

---

## 7. Voice Commands on Pi

`speech_recognition` with `recognize_google()` works on Pi but requires internet (sends audio to Google servers).

### Alternative: Offline Voice Commands with Vosk
```bash
pip install vosk

# Download small English model (~50MB)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip -d ~/vosk-model/
```

Replace the `_run` method in `voice/command_listener.py`:
```python
import json
from vosk import Model, KaldiRecognizer
import pyaudio

def _run(self):
    model = Model(os.path.expanduser("~/vosk-model/vosk-model-small-en-us-0.15"))
    rec = KaldiRecognizer(model, 16000)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=16000, input=True, frames_per_buffer=4096)

    while not self._stop_event.is_set():
        from audio.tts import is_speaking
        if is_speaking():
            time.sleep(0.1)
            continue

        data = stream.read(4096, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
            if text:
                self.logger.info("Heard: %s", text)
                normalized = self._normalize_command(text)
                if normalized:
                    self._commands.put(normalized)
```

| Feature | Google (current) | Vosk (offline) |
|---------|-----------------|----------------|
| Internet Required | Yes | No |
| Accuracy | Excellent | Very Good |
| Latency | 1-2s (network) | ~200ms |
| Model Size | N/A | ~50MB |
| Pi 5 CPU | Minimal | ~5% |

---

## 8. Camera Setup

### Pi Camera Module 3
```python
# In main.py, replace cv2.VideoCapture with picamera2
from picamera2 import Picamera2

cap = Picamera2()
config = cap.create_preview_configuration(main={"size": (640, 480)})
cap.configure(config)
cap.start()

# To read frames:
frame = cap.capture_array()
```

### USB Webcam
```python
# Works as-is with current OpenCV code
cap = cv2.VideoCapture(0)
# Reduce resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

---

## 9. Power & Thermal Management

### Cooling
```bash
# Check temperature
vcgencmd measure_temp

# If temperature > 80C, the Pi throttles CPU speed.
# Use an active cooler (fan) — essential for sustained AI workloads.
```

### Battery Runtime Estimates
| Power Bank | Runtime (approx) |
|-----------|------------------|
| 10,000mAh PD | ~1.5-2 hours |
| 20,000mAh PD | ~3-4 hours |
| 30,000mAh PD | ~5-6 hours |

Use a power bank that supports USB-C PD (Power Delivery) — the Pi 5 needs 5V/5A. Regular USB power banks may cause undervoltage warnings.

---

## 10. Auto-Start on Boot

Create a systemd service so the assistant starts automatically when Pi boots:

```bash
sudo nano /etc/systemd/system/vision-assist.service
```

```ini
[Unit]
Description=Vision Assist Wearable Assistant
After=network.target ollama.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Vision_Assist_2
ExecStart=/home/pi/Vision_Assist_2/venv/bin/python main.py
Restart=on-failure
RestartSec=5
Environment="GEMINI_API_KEY=your-key-here"
Environment="CLOUD_PROVIDER=gemini"
Environment="OLLAMA_NUM_THREADS=4"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable vision-assist
sudo systemctl start vision-assist

# Check logs
journalctl -u vision-assist -f
```

---

## 11. Deployment Checklist

### Pre-Deployment (Do on your PC)
- [ ] Export YOLOv8n to NCNN format (`model.export(format="ncnn", imgsz=320)`)
- [ ] Copy the NCNN model folder to Pi
- [ ] Update `vision.py` to use NCNN model path and 320px
- [ ] Make TTS cross-platform (add piper-tts for Linux)
- [ ] Test the full pipeline on PC with all changes

### On the Raspberry Pi
- [ ] Flash Raspberry Pi OS 64-bit (Bookworm)
- [ ] Install system dependencies (libopencv-dev, portaudio19-dev, etc.)
- [ ] Set up Python venv and install requirements.txt
- [ ] Install Ollama and pull qwen2.5:0.5b
- [ ] Install Piper TTS and download voice model
- [ ] Copy project files from PC
- [ ] Set up .env with GEMINI_API_KEY
- [ ] Connect camera, mic, and speaker
- [ ] Run `python main.py` and verify each mode
- [ ] Set up systemd auto-start service
- [ ] Test with a 20,000mAh power bank for portability

### Verification Tests
- [ ] Obstacle awareness detects objects and speaks warnings
- [ ] Scene description gives detailed Gemini descriptions after 4s
- [ ] Voice commands ("pause", "obstacle awareness") work between TTS speech
- [ ] System recovers gracefully from cloud API failures
- [ ] System runs for 30+ minutes without crashing
- [ ] Temperature stays below 80C with active cooling
