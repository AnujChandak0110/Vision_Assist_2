# Vision Assist Setup (Windows, Python 3.11)

This setup prepares the full wearable-assistant simulation pipeline:
- Voice command mode switching
- YOLOv8 scene understanding
- Local reasoning with Ollama
- Cloud fallback (Gemini/OpenAI)
- Priority TTS alerts

## 1) Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python --version
```

Expected: Python 3.11.x

## 2) Install dependencies

Runtime:

```powershell
python -m pip install -r requirements.txt
```

Testing:

```powershell
python -m pip install -r requirements-dev.txt
```

## 3) Install and prepare Ollama (local LLM)

Download Ollama:
https://ollama.com/download

Verify install:

```powershell
ollama --version
```

Pull local model:

```powershell
ollama pull phi3
```

Quick local check:

```powershell
ollama run phi3 "Give a short safety navigation instruction."
```

## 4) Configure cloud fallback (optional but recommended)

Create file `llm/.env`:

```env
CLOUD_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash

# Optional OpenAI fallback settings:
# OPENAI_API_KEY=your_openai_key
# OPENAI_MODEL=gpt-4o
```

`llm/cloud_api.py` auto-loads `llm/.env`.

## 5) Configure map-based navigation (optional)

For voice command `assistant navigate from <source> to <destination>` set:

```powershell
$env:GOOGLE_MAPS_API_KEY="your_maps_key"
```

## 6) Voice input notes

Voice command module uses `SpeechRecognition` with wake word: `assistant`.

Supported commands:
- `assistant start scene description mode`
- `assistant start navigation mode`
- `assistant switch to obstacle awareness`
- `assistant stop`
- `assistant pause`
- `assistant navigate from home to office`

If microphone backend is unavailable, the app still runs with camera + reasoning + TTS.

## 7) Run full system

```powershell
python main.py
```

## 8) Run QA tests

```powershell
python tests/test_pipeline.py
```

## 9) Safety and reliability checks

- Keep camera unobstructed and stable.
- Ensure internet for cloud/maps features.
- Keep local Ollama running for low-latency local reasoning.
- High-priority safety alerts can interrupt ongoing speech.
- Verify cloud keys are valid before field testing.
