"""Priority-aware offline TTS using pyttsx3 for assistive feedback."""

from __future__ import annotations

import logging
import queue
import threading
import time
from difflib import SequenceMatcher
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

import pyttsx3


class Priority(IntEnum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2


@dataclass
class SpeechMessage:
    priority: Priority
    text: str
    interrupt: bool
    category: str
    ts: float


@dataclass
class TTSConfig:
    rate: int = 175
    volume: float = 1.0
    dedup_window_seconds: float = 4.0
    cooldown_high: float = 0.3
    cooldown_medium: float = 1.5
    cooldown_low: float = 4.0
    similar_window_seconds: float = 6.0
    similar_ratio_threshold: float = 0.92
    global_min_gap_seconds: float = 0.9


class TextToSpeech:
    """Threaded speech engine with urgency-aware delivery controls."""

    def __init__(self, config: TTSConfig | None = None) -> None:
        self.config = config or TTSConfig()
        self.logger = self._build_logger()
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", self.config.rate)
        self._engine.setProperty("volume", self.config.volume)

        self._queue: queue.PriorityQueue[tuple[int, float, SpeechMessage]] = queue.PriorityQueue()
        self._lock = threading.Lock()

        self._last_text_ts: Dict[str, float] = {}
        self._last_category_ts: Dict[str, float] = {}
        self._recent_texts: List[Tuple[float, str, str]] = []
        self._last_any_ts: float = 0.0

        self._worker = threading.Thread(target=self._run, name="tts-thread", daemon=True)
        self._worker.start()

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("audio.tts")
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

    def _cooldown_for(self, priority: Priority) -> float:
        if priority == Priority.HIGH:
            return self.config.cooldown_high
        if priority == Priority.MEDIUM:
            return self.config.cooldown_medium
        return self.config.cooldown_low

    def _should_skip(self, msg: SpeechMessage) -> bool:
        now = time.time()
        dedup_key = f"{msg.category}:{msg.text.lower()}"

        if (now - self._last_any_ts) < self.config.global_min_gap_seconds:
            return True

        last_text = self._last_text_ts.get(dedup_key, 0.0)
        if (now - last_text) < self.config.dedup_window_seconds:
            return True

        candidate = " ".join(msg.text.lower().split())
        self._recent_texts = [
            (ts, cat, txt)
            for ts, cat, txt in self._recent_texts
            if (now - ts) <= self.config.similar_window_seconds
        ]
        for ts, cat, txt in self._recent_texts:
            _ = ts
            if cat != msg.category:
                continue
            if SequenceMatcher(None, candidate, txt).ratio() >= self.config.similar_ratio_threshold:
                return True

        last_cat = self._last_category_ts.get(msg.category, 0.0)
        if (now - last_cat) < self._cooldown_for(msg.priority):
            return True

        return False

    def _mark_sent(self, msg: SpeechMessage) -> None:
        now = time.time()
        dedup_key = f"{msg.category}:{msg.text.lower()}"
        self._last_text_ts[dedup_key] = now
        self._last_category_ts[msg.category] = now
        self._last_any_ts = now
        self._recent_texts.append((now, msg.category, " ".join(msg.text.lower().split())))

    def _clear_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return

    def speak(
        self,
        text: str,
        priority: str = "medium",
        interrupt: bool = False,
        category: str = "general",
    ) -> bool:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return False

        try:
            p = Priority[priority.upper()]
        except KeyError:
            p = Priority.MEDIUM

        msg = SpeechMessage(
            priority=p,
            text=cleaned,
            interrupt=interrupt,
            category=category.strip().lower() or "general",
            ts=time.time(),
        )

        with self._lock:
            if self._should_skip(msg):
                return False

            if msg.priority == Priority.HIGH and msg.interrupt:
                self._engine.stop()
                self._clear_pending()

            self._mark_sent(msg)
            self.logger.info("Queued speech: priority=%s category=%s text=%s", msg.priority.name.lower(), msg.category, msg.text)
            self._queue.put((int(msg.priority), msg.ts, msg))

        return True

    def _run(self) -> None:
        while True:
            _, _, msg = self._queue.get()
            try:
                self._engine.say(msg.text)
                self._engine.runAndWait()
            except Exception as exc:
                self.logger.error("TTS playback error: %s", exc)
                # Attempt lightweight recovery so future speech can continue.
                try:
                    self._engine = pyttsx3.init()
                    self._engine.setProperty("rate", self.config.rate)
                    self._engine.setProperty("volume", self.config.volume)
                except Exception as rexc:
                    self.logger.error("TTS engine recovery failed: %s", rexc)
            finally:
                self._queue.task_done()


_tts = TextToSpeech()


def speak(text: str, priority: str = "medium", interrupt: bool = False, category: str = "general") -> bool:
    """Queue speech asynchronously with priority and dedup controls."""
    return _tts.speak(text, priority=priority, interrupt=interrupt, category=category)


if __name__ == "__main__":
    speak("There is something right in front of you, please slow down.", priority="high", interrupt=True, category="safety")
    speak("You can shift a little to your left.", priority="medium", category="navigation")
    speak("It looks like a hallway with a chair on the side.", priority="low", category="scene")
    time.sleep(6)
