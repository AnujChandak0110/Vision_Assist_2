# Vision Assist 2 — Setup & Change Log (Windows, Python 3.11)

This setup prepares the full wearable-assistant pipeline for visually impaired users:
- Voice command mode switching with interrupt support
- YOLOv8n scene understanding (32 object classes)
- Local LLM reasoning with Ollama (llama3.2:1b)
- Cloud fallback with Google Gemini (gemini-3-flash-preview)
- Priority TTS alerts via Windows SAPI5 (multiprocessing)

---

## 1) Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python --version
```

Expected: Python 3.11.x

---

## 2) Install dependencies

```powershell
python -m pip install -r requirements.txt
```

For testing:

```powershell
python -m pip install -r requirements-dev.txt
```

---

## 3) Install and prepare Ollama (local LLM)

Download Ollama: https://ollama.com/download

```powershell
ollama --version

# Pull recommended edge model (fast, ~1.3GB)
ollama pull llama3.2:1b

# Or lighter model for slower hardware / Raspberry Pi
ollama pull qwen2.5:0.5b
```

Quick test:

```powershell
ollama run llama3.2:1b "Chair on center, near. What should a blind person do?"
```

> Note: phi3 was replaced with llama3.2:1b (3x faster). Timeout reduced from 25s to 8s.

---

## 4) Configure cloud API (Gemini)

Create or edit `.env` in the project root:

```env
CLOUD_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3-flash-preview
```

> Note: Model updated from gemini-2.5-flash to gemini-3-flash-preview for richer scene descriptions.
> Cloud API retry reduced from 3 to 1 attempt. Timeout reduced from 10s to 8s to prevent silence.

---

## 5) Configure map-based navigation (optional)

```powershell
$env:GOOGLE_MAPS_API_KEY="your_maps_key"
```

Voice command: `navigation` then describe route.

---

## 6) Voice commands

Voice commands work even while the assistant is speaking. You can interrupt at any time.

| Say | Action |
|-----|--------|
| `scene description` | Rich visual description of surroundings |
| `navigation` | Walking guidance mode |
| `obstacle awareness` | Continuous hazard detection |
| `pause` or `stop` | Pause all guidance |

> Note: Microphone is always active. During TTS playback, phrase limit drops to 2s for fast interrupt capture.

---

## 7) Run full system

```powershell
python main.py
```

---

## 8) Validate all pipeline fixes

```powershell
python validate_fixes.py
```

Expected: 22/22 checks passed.

---

## 9) Safety and reliability

- Keep camera unobstructed, at chest height facing forward.
- Ensure internet for Gemini cloud features.
- Keep Ollama running before starting: `ollama serve`
- High-priority hazard alerts interrupt all ongoing speech immediately.
- Cloud circuit breaker activates after 3 consecutive failures (20s cooldown).

---

## Changes Made (Session Log)

### Audio / TTS
- **Replaced pyttsx3 with win32com SAPI5** via dedicated multiprocessing worker to prevent Windows thread deadlocks.
- **IPC queue size increased** from 1 → 5 so hazard warnings don't block behind active audio playback.
- **Added `is_speaking()` export** from `audio/tts.py` so voice listener can check TTS state.

### Voice Command Listener
- **Microphone always stays active** — no longer muted during TTS playback.
- **Interrupt support**: During TTS speech, `phrase_time_limit` drops to 2s for quick command capture.
- Redundant `sleep + continue` skip logic removed.

### Vision / Object Detection
- **Expanded YOLO allowed classes** from 4 → 32 objects:
  - Added: car, bicycle, motorcycle, bus, truck, traffic light, stop sign, fire hydrant, dog, cat, horse, bench, couch, bed, dining table, toilet, laptop, cell phone, book, knife, scissors, potted plant, vase, and more.
- **Image size** remains 416px on PC, recommend 320px on Raspberry Pi.

### Scene Change Logic
- **`_is_major_scene_change`** now only triggers when object *classes* appear/disappear — not on every bounding box jitter. This prevents API 429 rate-limit spam.

### AI Pipeline
- **Async LLM worker**: LLM inference runs in a background thread so the camera loop is never blocked.
- **Keepalive messages suppressed** while LLM is actively generating a response.
- **Cloud circuit breaker**: `call_cloud_api` wrapped in try/except — HTTP exceptions now correctly trip the 20s cooldown instead of crashing the worker thread.

### Prompts (Redesigned for Blind Users)
All prompts rewritten to produce output a blind person can act on immediately:

**Before:**
- "There is an obstacle close on your left. Move a little to your right."
- "Move slowly and keep scanning for obstacles ahead."

**After:**
- "Chair close on your left. Step to your right."
- "Person directly ahead. Stop now."
- "Path ahead looks clear. Take two slow steps forward."

Gemini prompts now instruct the model to:
- Name every object (not just say "something")
- Use left/center/right positions
- Use near/medium/far distances in human terms (within 2 steps / 3-5 steps / beyond 5 steps)
- Give ONE actionable sentence per response
- Mention floor surfaces, steps, curbs, doors

### Cloud API (cloud_api.py)
- **Model**: gemini-2.5-flash → `gemini-3-flash-preview`
- **Obstacle awareness** now has its own dedicated prompt (shorter, safety-first)
- **Scene description tokens**: 60 → 200 maxOutputTokens
- **Scene description char limit**: 160 → 500 chars (allows 2-3 sentences)
- **False-positive failure detection fixed**: No longer treats valid responses containing "unavailable" as API failures.

### Sanitizer (main.py `_sanitize_instruction`)
- Scene description mode: allows multi-sentence output up to 60 words.
- Navigation/obstacle mode: strict single-sentence with expanded verb whitelist: `move, step, turn, keep, stop, pause, scan, continue, shift, slow, go, avoid, clear, watch, ahead, path`.

### RuntimeConfig
- `inference_min_interval_sec`: 0.35 (PC default, use 1.0 for Raspberry Pi)
- `speech_keepalive_interval_sec`: 12s
- `cloud_min_interval_sec`: 4s

---

## Raspberry Pi 5 Deployment

See `implementation.md` for full Raspberry Pi deployment guide including:
- Hardware requirements and recommended accessories
- Piper TTS setup (replaces SAPI5 on Linux)
- YOLO NCNN export (2 FPS → 6-8 FPS on Pi CPU)
- Ollama model selection for edge devices
- Pi-optimized RuntimeConfig values
- Auto-start systemd service
- Full deployment checklist
