"""
Voice command listener for persistent mode control.

Commands supported:
- Start scene description mode
- Start navigation mode
- Switch to obstacle awareness
- Stop / Pause

Includes wake-word gate (default: "assistant").
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import speech_recognition as sr
except Exception:  # pragma: no cover - optional runtime dependency behavior
    sr = None


@dataclass
class VoiceConfig:
    wake_word: str = "assistant"
    phrase_time_limit: float = 4.0
    listen_timeout: float = 1.0
    energy_threshold: int = 300
    require_wake_word: bool = False


class VoiceCommandListener:
    """Background voice listener that emits normalized command strings."""

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self.config = config or VoiceConfig()
        self.logger = self._build_logger()

        self._stop_event = threading.Event()
        self._commands: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="voice-thread")

        self._recognizer = None
        self._microphone = None
        if sr is not None:
            try:
                self._recognizer = sr.Recognizer()
                self._recognizer.energy_threshold = self.config.energy_threshold
                self._recognizer.dynamic_energy_threshold = True
                self._microphone = sr.Microphone()
            except Exception as exc:
                self.logger.warning("Microphone init failed: %s", exc)

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("voice.listener")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
            )
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def get_command_nowait(self) -> Optional[str]:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def is_available(self) -> bool:
        return self._recognizer is not None and self._microphone is not None

    def inject_command(self, text: str) -> None:
        """Manual command injection for testing/simulation."""
        normalized = self._normalize_command(text)
        if normalized:
            self._commands.put(normalized)

    def _normalize_command(self, text: str) -> Optional[str]:
        lowered = " ".join(text.strip().lower().split())
        if not lowered:
            return None

        wake = self.config.wake_word.strip().lower()
        if self.config.require_wake_word and wake and wake not in lowered:
            return None

        # Remove wake-word prefix if present.
        if wake and wake in lowered:
            lowered = lowered.replace(wake, "", 1).strip(" ,")

        if "scene" in lowered and "description" in lowered:
            return "scene_description"
        if "navigation" in lowered:
            return "navigation"
        if "obstacle" in lowered:
            return "obstacle_awareness"
        if lowered in {"stop", "pause"} or "stop" in lowered or "pause" in lowered:
            return "pause"

        # Optional map navigation command:
        # assistant navigate from home to office
        if lowered.startswith("navigate from ") and " to " in lowered:
            return f"route:{lowered}"

        if lowered.startswith("switch to "):
            target = lowered.replace("switch to ", "", 1).strip()
            if target:
                return f"app_switch:{target}"

        return None

    def _run(self) -> None:
        if self._recognizer is None or self._microphone is None:
            self.logger.warning("SpeechRecognition unavailable; voice commands disabled (use inject_command).")
            while not self._stop_event.is_set():
                time.sleep(0.25)
            return

        self.logger.info("Voice listener started with wake word '%s'.", self.config.wake_word)
        with self._microphone as source:
            try:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
            except Exception as exc:
                self.logger.warning("Ambient noise calibration failed: %s", exc)

            while not self._stop_event.is_set():
                # During TTS playback, use a shorter phrase limit so we quickly
                # check for interrupt commands without long blocking waits.
                # The mic stays ACTIVE so the user can always interrupt.
                try:
                    from audio.tts import is_speaking
                    tts_active = is_speaking()
                except ImportError:
                    tts_active = False

                phrase_limit = 2.0 if tts_active else self.config.phrase_time_limit

                try:
                    audio = self._recognizer.listen(
                        source,
                        timeout=self.config.listen_timeout,
                        phrase_time_limit=phrase_limit,
                    )
                except Exception:
                    continue

                try:
                    text = self._recognizer.recognize_google(audio)
                except Exception:
                    continue

                self.logger.info("Heard: %s", text)

                normalized = self._normalize_command(text)
                if normalized:
                    self.logger.info("Voice command: %s", normalized)
                    self._commands.put(normalized)

        self.logger.info("Voice listener stopped.")
