"""
Local Ollama integration for edge AI reasoning.

Features:
- Endpoint: http://localhost:11434/api/generate
- Model: phi3 (configurable)
- Timeout: 5 seconds (configurable)
- Prompt construction for blind navigation assistant context
- Graceful timeout/failure handling
- Basic logging and response trimming
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict

import requests
from requests import RequestException
from requests.exceptions import Timeout

DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_TIMEOUT_SECONDS = 25
DEFAULT_MAX_RESPONSE_CHARS = 160


@dataclass
class OllamaConfig:
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("llm.ollama_local")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


LOGGER = _get_logger()


def build_prompt(scene_data: Dict[str, Any]) -> str:
    """
    Build a focused prompt for blind navigation guidance or scene description.

    Expected scene_data is any JSON-serializable dictionary containing
    object detections, positions, distances, and environment cues.
    """
    compact_scene = json.dumps(scene_data, ensure_ascii=True, separators=(",", ":"))
    mode = scene_data.get("mode", "navigation")

    if mode == "scene_description":
        return (
            "You are an assistant for a blind user. "
            "Return exactly one short sentence describing the scene in front of the user. "
            "Prioritize naming objects, their positions, and distances. "
            "Use plain language, maximum 14 words. "
            "No explanation, no reasoning. "
            "Never suggest approaching or following people. "
            "If uncertain, say: Pause and scan around. "
            f"Scene data: {compact_scene}"
        )

    return (
        "You are a blind navigation assistant. "
        "Return exactly one short sentence for immediate safe movement. "
        "Use plain language, maximum 14 words. "
        "No explanation, no reasoning, no meta text, no multiple options. "
        "Never suggest approaching or following people. "
        "If uncertain, say: Pause and scan around before moving. "
        f"Scene data: {compact_scene}"
    )


def _trim_response(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.strip().split())
    # Keep a single sentence to avoid long or malformed spoken output.
    for sep in (".", "!", "?"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip()
            break

    if len(cleaned) <= max_chars:
        return cleaned

    # Keep the output concise for realtime speech feedback.
    return cleaned[: max_chars - 3].rstrip() + "..."


def query_ollama(scene_data: Dict[str, Any], config: OllamaConfig | None = None) -> str:
    """
    Query Ollama and return a short response text.

    Handles timeout/failure gracefully by returning a safe fallback instruction.
    """
    cfg = config or OllamaConfig()
    payload = {
        "model": cfg.model,
        "prompt": build_prompt(scene_data),
        "stream": False,
        "keep_alive": -1,  # Keep the model loaded in memory indefinitely to prevent high latency
    }

    try:
        LOGGER.info("Sending request to Ollama model=%s endpoint=%s", cfg.model, cfg.endpoint)
        response = requests.post(
            cfg.endpoint,
            json=payload,
            timeout=cfg.timeout_seconds,
        )
        response.raise_for_status()

        body = response.json()
        text = str(body.get("response", "")).strip()
        if not text:
            LOGGER.warning("Empty response from Ollama.")
            return "Pause and scan surroundings again."

        return _trim_response(text, cfg.max_response_chars)

    except Timeout:
        LOGGER.warning("Ollama request timed out after %ss", cfg.timeout_seconds)
        return "No response yet. Stop safely and scan again."
    except RequestException as exc:
        LOGGER.error("Ollama request failed: %s", exc)
        return "Reasoning unavailable. Move slowly and scan again."
    except ValueError as exc:
        LOGGER.error("Invalid JSON from Ollama: %s", exc)
        return "Unclear guidance. Pause and scan surroundings."


if __name__ == "__main__":
    sample_scene = {
        "detections": [
            {"label": "person", "confidence": 0.91, "position": "center", "distance": "near"},
            {"label": "chair", "confidence": 0.77, "position": "left", "distance": "medium"},
        ]
    }

    result = query_ollama(sample_scene)
    print(result)
