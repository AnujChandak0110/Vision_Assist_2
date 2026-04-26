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
import re
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

    @staticmethod
    def _compact_lower(text: str) -> str:
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _contains_any(text: str, phrases: list[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    def _strip_wake_word(self, text: str) -> str:
        wake = self.config.wake_word.strip().lower()
        if not wake:
            return text
        if text.startswith(wake + " "):
            return text[len(wake):].strip(" ,")
        return text.replace(wake, "", 1).strip(" ,") if wake in text else text

    def _parse_route_command(self, text: str) -> Optional[str]:
        from_to = re.search(
            r"(?:navigate|guid(e|ing)|take me|walk|go|route)\s+(?:me\s+)?from\s+(.+?)\s+to\s+(.+)$",
            text,
        )
        if from_to:
            origin = from_to.group(2).strip()
            destination = from_to.group(3).strip()
            if origin and destination:
                return f"route:navigate from {origin} to {destination}"

        to_only = re.search(
            r"(?:navigate|guid(e|ing)|take me|walk|go|route)\s+(?:me\s+)?to\s+(.+)$",
            text,
        )
        if to_only:
            destination = to_only.group(2).strip()
            if destination:
                return f"route_to:{destination}"

        return None

    def _normalize_command(self, text: str) -> Optional[str]:
        lowered = self._compact_lower(text)
        if not lowered:
            return None

        wake = self.config.wake_word.strip().lower()
        if self.config.require_wake_word and wake and wake not in lowered:
            return None

        lowered = self._strip_wake_word(lowered)
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = self._compact_lower(lowered)
        if not lowered:
            return None

        pause_phrases = ["stop", "pause", "hold on", "wait", "be quiet", "silence"]
        if self._contains_any(lowered, pause_phrases):
            return "pause"

        resume_phrases = ["resume", "continue", "go on", "keep going", "start again"]
        if self._contains_any(lowered, resume_phrases):
            return "resume"

        rescan_phrases = [
            "scan again",
            "rescan",
            "look around",
            "check around",
            "full scan",
            "scan the room",
        ]
        if self._contains_any(lowered, rescan_phrases):
            return "rescan"

        route_cmd = self._parse_route_command(lowered)
        if route_cmd:
            return route_cmd

        obstacle_phrases = [
            "obstacle",
            "watch out",
            "hazard",
            "safety mode",
            "avoid collisions",
            "collision",
        ]
        if self._contains_any(lowered, obstacle_phrases):
            return "obstacle_awareness"

        navigation_phrases = [
            "navigation",
            "guide me",
            "help me walk",
            "way out",
            "exit",
            "lead me",
            "take me",
        ]
        if self._contains_any(lowered, navigation_phrases):
            return "navigation"

        scene_phrases = [
            "scene description",
            "describe",
            "what is around me",
            "what do you see",
            "surroundings",
            "environment",
            "look mode",
        ]
        if self._contains_any(lowered, scene_phrases):
            return "scene_description"

        if lowered.startswith("switch to "):
            target = lowered.replace("switch to ", "", 1).strip()
            if target:
                return f"app_switch:{target}"

        if lowered.startswith("open "):
            target = lowered.replace("open ", "", 1).replace(" mode", "").strip()
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
                # During TTS playback, use shorter windows so user interrupt
                # commands are captured quickly.
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
