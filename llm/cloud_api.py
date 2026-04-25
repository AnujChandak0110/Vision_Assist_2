"""
Fallback cloud reasoning for navigation.

Supports either:
- OpenAI GPT-4o
- Google Gemini

Provider is selected via CLOUD_PROVIDER environment variable:
- "openai" (default)
- "gemini"
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Final

import requests
from requests import RequestException

OPENAI_ENDPOINT: Final[str] = "https://api.openai.com/v1/chat/completions"
GEMINI_ENDPOINT_TEMPLATE: Final[str] = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

DEFAULT_OPENAI_MODEL: Final[str] = "gpt-4o"
DEFAULT_GEMINI_MODEL: Final[str] = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS: Final[int] = 10
DEFAULT_MAX_CHARS: Final[int] = 160
MAX_RETRIES: Final[int] = 3
BACKOFF_BASE_SECONDS: Final[float] = 1.0
BACKOFF_MAX_SECONDS: Final[float] = 8.0
RETRYABLE_STATUS_CODES: Final[set[int]] = {429, 500, 502, 503, 504}

ALLOWED_REASONS: Final[set[str]] = {
    "low_confidence",
    "persistence",
    "long_term_context",
    "scene_description_accuracy",
}


def _load_local_env_files() -> None:
    """Load key=value pairs from local .env files if present."""
    env_paths = (
        Path(__file__).with_name(".env"),
        Path(__file__).resolve().parent.parent / ".env",
    )

    for env_path in env_paths:
        if not env_path.exists() or not env_path.is_file():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue

            # Do not override shell-exported values.
            os.environ.setdefault(key, value)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("llm.cloud_api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


LOGGER = _get_logger()
_load_local_env_files()


def _trim_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    cleaned = " ".join(text.strip().split())
    # Keep first sentence only for stable speech output.
    for sep in (".", "!", "?"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip()
            break

    if len(cleaned) <= max_chars:
        return cleaned

    snippet = cleaned[:max_chars].rstrip()

    # Prefer natural sentence boundaries to avoid partial spoken output.
    for mark in (".", "!", "?"):
        idx = snippet.rfind(mark)
        if idx >= 24:
            return snippet[: idx + 1].strip()

    cut = snippet.rfind(" ")
    if cut >= 24:
        return snippet[:cut].strip()

    return snippet


def build_prompt(scene: str, reason: str, mode: str = "navigation") -> str:
    """
    Build a short, safety-focused prompt based on the mode.

    reason should be:
    - low_confidence
    - persistence
    """
    reason_normalized = reason.strip().lower()
    if reason_normalized not in ALLOWED_REASONS:
        reason_normalized = "low_confidence"

    if mode == "scene_description":
        return (
            "You are an assistant for a blind user. "
            "Return exactly one short sentence describing the scene in front of the user. "
            "Prioritize naming objects, their positions, and distances. "
            "No explanation, no reasoning, max 14 words. "
            "Never suggest approaching or following people. "
            f"Scene: {scene.strip()}"
        )

    return (
        "You are a navigation assistant for a blind user. "
        "Return exactly one short, actionable movement instruction for the next few seconds. "
        "Prioritize safety and obstacle avoidance. "
        "No explanation, no reasoning, no meta text, max 14 words. "
        "Never suggest approaching or following people. "
        f"Trigger reason: {reason_normalized}. "
        f"Scene: {scene.strip()}"
    )


def _request_with_backoff(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout: int,
    params: dict[str, str] | None = None,
) -> requests.Response:
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                params=params,
            )

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                delay = min(BACKOFF_BASE_SECONDS * (2**attempt), BACKOFF_MAX_SECONDS)
                delay += random.uniform(0.0, 0.4)
                LOGGER.warning(
                    "Cloud retryable status=%s attempt=%s/%s; backing off for %.2fs",
                    response.status_code,
                    attempt + 1,
                    MAX_RETRIES + 1,
                    delay,
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response
        except RequestException as exc:
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break

            delay = min(BACKOFF_BASE_SECONDS * (2**attempt), BACKOFF_MAX_SECONDS)
            delay += random.uniform(0.0, 0.4)
            LOGGER.warning(
                "Cloud request exception on attempt=%s/%s; backing off for %.2fs",
                attempt + 1,
                MAX_RETRIES + 1,
                delay,
            )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Cloud request failed without exception context")


def _call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return concise navigation instructions only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 60,
    }

    response = _request_with_backoff(
        url=OPENAI_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    body = response.json()

    choices = body.get("choices", [])
    if not choices:
        return "Stop briefly, scan around, then move slowly forward."

    content = choices[0].get("message", {}).get("content", "")
    return str(content).strip()


def _call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    endpoint = GEMINI_ENDPOINT_TEMPLATE.format(model=model)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 60,
        },
    }

    response = _request_with_backoff(
        url=endpoint,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        payload=payload,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    body = response.json()

    candidates = body.get("candidates", [])
    if not candidates:
        return "Stop briefly, scan around, then move slowly forward."

    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return "Stop briefly, scan around, then move slowly forward."

    text = parts[0].get("text", "")
    return str(text).strip()


def call_cloud_api(scene: str, reason: str, mode: str = "navigation") -> str:
    """
    Call selected cloud model and return short navigation text.

    Environment variables:
    - CLOUD_PROVIDER=openai|gemini
    - OPENAI_API_KEY (+ optional OPENAI_MODEL)
    - GEMINI_API_KEY (+ optional GEMINI_MODEL)
    """
    prompt = build_prompt(scene, reason, mode)
    provider = os.getenv("CLOUD_PROVIDER", "gemini").strip().lower()

    try:
        if provider == "gemini":
            LOGGER.info("Using cloud provider: gemini")
            result = _call_gemini(prompt)
        else:
            LOGGER.info("Using cloud provider: openai")
            result = _call_openai(prompt)

        return _trim_text(result)

    except (RequestException, RuntimeError, ValueError) as exc:
        LOGGER.error("Cloud API call failed: %s", exc)
        return "Cloud guidance unavailable. Stop safely and rescan surroundings."


if __name__ == "__main__":
    sample_scene = "person at left, near; chair at center, far"
    print(call_cloud_api(sample_scene, "low_confidence"))
