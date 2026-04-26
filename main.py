"""Real-time wearable assistant pipeline with voice-driven multi-mode operation."""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import os
import platform

import cv2

from audio.tts import speak, start_tts
from decision.decision_engine import compute_confidence
from llm.cloud_api import call_cloud_api
from llm.ollama_local import query_ollama
from utils.frame_change import FrameChangeConfig, FrameChangeDetector
from utils.maps_navigation import get_next_navigation_instruction
from utils.modes import AssistantMode
from utils.scene_formatter import format_scene
from vision.vision import DetectorConfig, VisionDetector
from voice.command_listener import VoiceCommandListener

Detection = Dict[str, Any]


def _load_env_defaults() -> None:
    """Load CAMERA_INDEX and other env vars from .env into RuntimeConfig defaults."""
    from pathlib import Path
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)

_load_env_defaults()


@dataclass
class RuntimeConfig:
    camera_index: int = int(os.environ.get("CAMERA_INDEX", "0"))
    frame_stride: int = 3
    queue_size: int = 2
    detector_debug: bool = False
    min_recompute_interval_sec: float = 2.0
    nav_instruction_interval_sec: float = 5.0
    long_term_cloud_interval_sec: float = 12.0
    mode_status_interval_sec: float = 8.0
    inference_min_interval_sec: float = 0.35
    cloud_min_interval_sec: float = 8.0
    cloud_failure_cooldown_sec: float = 30.0
    cloud_breaker_failure_threshold: int = 2
    scene_stale_refresh_sec: float = 6.0
    speech_keepalive_interval_sec: float = 12.0
    cloud_low_confidence_threshold: float = 0.45
    cloud_high_detail_object_threshold: int = 5
    scan_hold_seconds: float = 2.8
    navigation_scan_stale_sec: float = 90.0
    scan_heading_labels: Tuple[str, ...] = (
        "12 o'clock",
        "2 o'clock",
        "4 o'clock",
        "6 o'clock",
        "8 o'clock",
        "10 o'clock",
    )


class RealTimeAISystem:
    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self.logger = self._build_logger()

        self.stop_event = threading.Event()
        self.frame_queue: queue.Queue[Any] = queue.Queue(maxsize=self.config.queue_size)

        self.detector = VisionDetector(
            DetectorConfig(
                image_size=416,
                confidence_threshold=0.5,
                skip_frames=0,
                debug=self.config.detector_debug,
            )
        )
        self.frame_change = FrameChangeDetector(FrameChangeConfig())

        self.voice_listener = VoiceCommandListener()

        self._state_lock = threading.Lock()
        self._llm_lock = threading.Lock()
        self._is_llm_running = False
        self._mode = AssistantMode.SCENE_DESCRIPTION
        self._previous_mode = AssistantMode.SCENE_DESCRIPTION

        self._route_origin: Optional[str] = None
        self._route_destination: Optional[str] = None
        self._last_nav_prompt_ts = 0.0
        self._last_mode_status_ts = 0.0
        self._last_cloud_context_ts: Dict[str, float] = {}
        self._last_inference_step_ts = 0.0

        self._last_cloud_attempt_ts = 0.0
        self._cloud_failures = 0
        self._cloud_breaker_until_ts = 0.0
        self._last_cloud_response_by_mode: Dict[str, str] = {}
        self._last_cloud_signature_by_mode: Dict[str, str] = {}

        self._cached_detections: List[Detection] = []
        self._cached_enriched: List[Detection] = []
        self._cached_scene_text: str = "no objects detected"
        self._last_detect_ts = 0.0
        self._last_semantic_scene: Set[Tuple[str, str, str]] = set()
        self._last_major_change_ts = 0.0
        self._last_spoken_ts = 0.0

        self._scan_active = False
        self._scan_purpose = "startup"
        self._scan_heading_index = 0
        self._scan_prompted_index = -1
        self._scan_step_started_ts = 0.0
        self._scan_snapshots: Dict[str, List[Tuple[str, str, str]]] = {}
        self._spatial_memory: Dict[str, List[Tuple[str, str, str]]] = {}
        self._scan_last_completed_ts = 0.0

        self._camera_thread = threading.Thread(target=self._camera_loop, name="camera-thread", daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, name="inference-thread", daemon=True)

    @staticmethod
    def _build_logger() -> logging.Logger:
        logger = logging.getLogger("main")
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

    def start(self) -> None:
        self.logger.info("Starting wearable assistant pipeline.")
        start_tts()
        speak(
            "I am with you and ready to guide. Say scene mode, navigation, or obstacle mode anytime.",
            priority="medium",
            category="system",
        )

        if self.voice_listener.is_available():
            speak("I am listening. You can interrupt me anytime with assistant.", priority="low", category="system")
        else:
            speak(
                "Microphone not found. Camera guidance is still active.",
                priority="medium",
                category="system",
            )

        self.voice_listener.start()
        self._camera_thread.start()
        self._inference_thread.start()
        self._start_environment_scan("startup", force=True)

        try:
            while not self.stop_event.is_set():
                self._poll_voice_commands()
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received. Stopping system.")
            self.stop()

    def stop(self) -> None:
        self.stop_event.set()
        self.voice_listener.stop()
        self._camera_thread.join(timeout=2.0)
        self._inference_thread.join(timeout=2.0)
        self.logger.info("System stopped.")

    def _poll_voice_commands(self) -> None:
        while True:
            cmd = self.voice_listener.get_command_nowait()
            if not cmd:
                break
            self._apply_voice_command(cmd)

    def _apply_voice_command(self, cmd: str) -> None:
        if cmd == "rescan":
            purpose = "navigation" if self._mode == AssistantMode.NAVIGATION else "startup"
            self._start_environment_scan(purpose, force=True)
            return

        if cmd == "resume":
            with self._state_lock:
                if self._mode == AssistantMode.PAUSED:
                    self._mode = self._previous_mode
            speak("Back with you. Continuing guidance now.", priority="medium", category="system")
            return

        if cmd.startswith("app_switch:"):
            app_name = cmd.split(":", 1)[1].strip()
            if app_name:
                speak(
                    f"I heard switch to {app_name}, but app switching is not connected yet. Staying in current assist mode.",
                    priority="low",
                    category="system",
                )
            return

        if cmd.startswith("route:"):
            route_payload = cmd.split(":", 1)[1]
            m = re.search(r"navigate from (.+) to (.+)", route_payload)
            if not m:
                speak("I did not catch the route details. Please repeat source and destination.", category="system")
                return

            origin = m.group(1).strip()
            destination = m.group(2).strip()
            with self._state_lock:
                self._route_origin = origin
                self._route_destination = destination
                self._mode = AssistantMode.NAVIGATION
                self._previous_mode = AssistantMode.NAVIGATION

            speak(
                f"Navigation is on. I will guide you from {origin} to {destination}.",
                priority="medium",
                category="navigation",
            )
            self._start_environment_scan("navigation", force=True)
            return

        if cmd.startswith("route_to:"):
            destination = cmd.split(":", 1)[1].strip()
            if not destination:
                speak("I heard navigation, but not the destination. Please say it again.", category="system")
                return

            with self._state_lock:
                self._route_origin = "current location"
                self._route_destination = destination
                self._mode = AssistantMode.NAVIGATION
                self._previous_mode = AssistantMode.NAVIGATION

            speak(
                f"Okay, we will head to {destination}. First, let us scan the full space around you.",
                priority="medium",
                category="navigation",
            )
            self._start_environment_scan("navigation", force=True)
            return

        mapping = {
            "scene_description": AssistantMode.SCENE_DESCRIPTION,
            "navigation": AssistantMode.NAVIGATION,
            "obstacle_awareness": AssistantMode.OBSTACLE_AWARENESS,
            "pause": AssistantMode.PAUSED,
        }
        mode = mapping.get(cmd)
        if mode is None:
            return

        with self._state_lock:
            self._mode = mode
            if mode != AssistantMode.PAUSED:
                self._previous_mode = mode

        if mode == AssistantMode.PAUSED:
            speak("Pausing guidance. Say assistant and a mode command to continue.", priority="medium", category="system")
        else:
            readable = mode.value.replace("_", " ")
            speak(f"Okay, {readable} mode is active.", priority="medium", category="system")
            if mode == AssistantMode.NAVIGATION and self._scan_is_stale():
                self._start_environment_scan("navigation", force=True)

    def _camera_loop(self) -> None:
        # Use DirectShow on Windows for faster USB webcam init and stable reads
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.config.camera_index, backend)
        if not cap.isOpened():
            # Fallback: try without backend hint
            cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            self.logger.error("Unable to open camera index %s", self.config.camera_index)
            speak("Camera unavailable. Please check camera connection.", priority="high", interrupt=True, category="safety")
            self.stop_event.set()
            return

        # Set camera resolution explicitly for consistent bbox coordinates
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.logger.info("Camera thread started (640x480).")
        consecutive_failures = 0
        frame_interval = 1.0 / 15.0   # Target 15 FPS — no need to spin faster

        while not self.stop_event.is_set():
            t0 = time.time()
            ok, frame = cap.read()

            if not ok:
                consecutive_failures += 1
                if consecutive_failures == 5:
                    self.logger.warning("Camera: 5 consecutive read failures. Check camera connection.")
                    speak("Camera signal lost. Trying to recover.", priority="high", interrupt=True, category="safety")
                if consecutive_failures > 30:
                    self.logger.error("Camera: too many failures, stopping.")
                    self.stop_event.set()
                    break
                time.sleep(0.1)
                continue

            consecutive_failures = 0

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

            # Pace the camera loop to avoid CPU burn
            elapsed = time.time() - t0
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        cap.release()
        self.logger.info("Camera thread stopped.")

    def _async_llm_worker(self, mode: AssistantMode, scene_text: str, enriched: List[Detection], avg_conf: float, recompute: bool, scene_signature: str, major_scene_change: bool, complex_environment: bool, cloud_needed: bool) -> None:
        try:
            response_text = ""
            source = ""
            priority = "low"
            interrupt = False

            if mode == AssistantMode.OBSTACLE_AWARENESS:
                if recompute and major_scene_change:
                    response_text = self._local_fast_response(mode, scene_text, enriched, avg_conf)
                    source = "local_llm"
                    priority = "medium"

                cloud_text = self._call_cloud_with_guardrails(mode, scene_text, "long_term_context", scene_signature, recompute, major_scene_change, complex_environment, cloud_needed)
                if cloud_text:
                    response_text = cloud_text
                    source = "cloud"

            elif mode == AssistantMode.SCENE_DESCRIPTION:
                response_text, source = self._scene_cloud_response(scene_text, enriched, avg_conf, mode, recompute, scene_signature, major_scene_change, complex_environment, cloud_needed)
                priority = "low"

            elif mode == AssistantMode.NAVIGATION:
                response_text, source, priority, interrupt = self._navigation_response(scene_text, enriched, avg_conf, recompute, scene_signature, major_scene_change, complex_environment, cloud_needed)

            elif mode in {AssistantMode.OBJECT_FINDER, AssistantMode.TEXT_READING}:
                if recompute and major_scene_change:
                    scene_data = {
                        "mode": mode.value,
                        "scene": scene_text,
                        "detections": enriched,
                        "instruction_style": "one short safe instruction, no reasoning",
                    }
                    from llm.ollama_local import query_ollama
                    response_text = query_ollama(scene_data)
                    source = "local_llm"
                    priority = "low"

            if response_text and source != "rule":
                response_text = self._sanitize_instruction(response_text, mode, scene_text)

            if response_text:
                category = "safety" if priority == "high" else ("navigation" if mode == AssistantMode.NAVIGATION else "scene")
                queued = speak(response_text, priority=priority, interrupt=interrupt, category=category)
                if queued:
                    self._last_spoken_ts = time.time()
                self.logger.info("async_llm: mode=%s source=%s conf=%.3f queued=%s", mode.value, source, avg_conf, queued)
        except Exception as exc:
            self.logger.error("Async LLM error: %s", exc)
        finally:
            with self._llm_lock:
                self._is_llm_running = False

    def _enrich_detections(self, detections: List[Detection]) -> List[Detection]:
        enriched: List[Detection] = []
        for det in detections:
            bbox = det.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue

            entry = dict(det)
            entry["position"] = self.detector.get_position(bbox)
            entry["distance"] = self.detector.estimate_distance(bbox)
            enriched.append(entry)
        return enriched

    def _scan_is_stale(self) -> bool:
        if not self._spatial_memory:
            return True
        return (time.time() - self._scan_last_completed_ts) >= self.config.navigation_scan_stale_sec

    def _start_environment_scan(self, purpose: str, force: bool = False) -> None:
        if self._scan_active and not force:
            return

        if purpose == "navigation" and not force and not self._scan_is_stale():
            return

        self._scan_active = True
        self._scan_purpose = purpose
        self._scan_heading_index = 0
        self._scan_prompted_index = -1
        self._scan_step_started_ts = 0.0
        self._scan_snapshots = {}

        if purpose == "navigation":
            speak(
                "Before we move, let us do a quick full scan. Turn slowly with me so I can map safe directions.",
                priority="medium",
                category="navigation",
            )
        else:
            speak(
                "Let us begin fresh. Please turn slowly in a full circle while I learn your surroundings.",
                priority="medium",
                category="system",
            )

    @staticmethod
    def _compact_scan_snapshot(enriched: List[Detection]) -> List[Tuple[str, str, str]]:
        rank = {"near": 0, "medium": 1, "far": 2}
        compact: List[Tuple[str, str, str]] = []
        seen: Set[Tuple[str, str, str]] = set()

        sorted_enriched = sorted(
            enriched,
            key=lambda d: rank.get(str(d.get("distance", "far")).lower(), 2),
        )
        for det in sorted_enriched:
            label = str(det.get("label", "object")).strip().lower()
            position = str(det.get("position", "center")).strip().lower()
            distance = str(det.get("distance", "far")).strip().lower()
            key = (label, position, distance)
            if not label or key in seen:
                continue
            seen.add(key)
            compact.append(key)
            if len(compact) >= 4:
                break

        return compact

    def _scan_step_prompt(self, heading: str, index: int) -> str:
        if index == 0:
            return f"Face {heading} and hold for a moment. I am checking this direction now."
        return f"Good. Now turn gently to {heading} and hold there for one second."

    @staticmethod
    def _direction_risk(snapshot: List[Tuple[str, str, str]]) -> int:
        distance_risk = {"near": 5, "medium": 3, "far": 1}
        risk = 0
        for _, position, distance in snapshot:
            risk += distance_risk.get(distance, 2)
            if position == "center":
                risk += 1
        return risk

    def _best_heading_from_memory(self) -> Optional[str]:
        if not self._spatial_memory:
            return None

        best_heading = None
        best_score = 10**6
        for heading in self.config.scan_heading_labels:
            snapshot = self._spatial_memory.get(heading, [])
            score = self._direction_risk(snapshot)
            if score < best_score:
                best_score = score
                best_heading = heading
        return best_heading

    def _scan_memory_summary_sentence(self) -> str:
        if not self._spatial_memory:
            return "I still need a little more scan data."

        best_heading = self._best_heading_from_memory()
        if not best_heading:
            return "I have a rough layout now."

        best_snapshot = self._spatial_memory.get(best_heading, [])
        if not best_snapshot:
            return f"The clearest space is around {best_heading}."

        top_label = best_snapshot[0][0]
        return f"The cleanest direction is around {best_heading}, with {top_label} off to the side."

    def _navigation_hint_from_memory(self) -> str:
        best_heading = self._best_heading_from_memory()
        if not best_heading:
            return "Take two short steps forward and keep your cane sweeping left to right."
        return f"From the scan, the safest opening is near {best_heading}. Turn slightly that way and take three small steps."

    def _update_environment_scan(self, enriched: List[Detection], recompute: bool) -> Optional[str]:
        if not self._scan_active:
            return None

        headings = self.config.scan_heading_labels
        if self._scan_heading_index >= len(headings):
            return None

        now = time.time()
        heading = headings[self._scan_heading_index]

        if self._scan_prompted_index != self._scan_heading_index:
            self._scan_prompted_index = self._scan_heading_index
            self._scan_step_started_ts = now
            return self._scan_step_prompt(heading, self._scan_heading_index)

        if recompute and heading not in self._scan_snapshots:
            self._scan_snapshots[heading] = self._compact_scan_snapshot(enriched)

        if (now - self._scan_step_started_ts) < self.config.scan_hold_seconds:
            return None

        if heading not in self._scan_snapshots:
            self._scan_snapshots[heading] = self._compact_scan_snapshot(enriched)

        self._scan_heading_index += 1
        self._scan_prompted_index = -1

        if self._scan_heading_index < len(headings):
            return None

        self._spatial_memory = dict(self._scan_snapshots)
        self._scan_last_completed_ts = time.time()
        self._scan_active = False
        summary = self._scan_memory_summary_sentence()

        if self._scan_purpose == "navigation":
            return f"Great, I mapped your space. {summary} I will guide you step by step now."
        return f"Thanks for turning. I have a better picture now. {summary}"

    def _should_recompute_detections(self, frame: Any) -> bool:
        if not self._cached_detections:
            return True

        now = time.time()
        changed = self.frame_change.has_significant_change(frame)
        min_due = (now - self._last_detect_ts) >= self.config.min_recompute_interval_sec
        stale_refresh_due = (now - self._last_detect_ts) >= self.config.scene_stale_refresh_sec

        # Recompute only on visual change with minimum pacing, plus sparse stale refresh.
        return (changed and min_due) or stale_refresh_due

    def _semantic_scene_signature(self, enriched: List[Detection]) -> Set[Tuple[str, str, str]]:
        sig: Set[Tuple[str, str, str]] = set()
        for det in enriched:
            label = str(det.get("label", "")).strip().lower()
            pos = str(det.get("position", "")).strip().lower()
            dist = str(det.get("distance", "")).strip().lower()
            if label:
                sig.add((label, pos, dist))
        return sig

    @staticmethod
    def _is_major_scene_change(prev_sig: Set[Tuple[str, str, str]], curr_sig: Set[Tuple[str, str, str]]) -> bool:
        if not prev_sig and curr_sig:
            return True

        # Only trigger major changes when object classes (labels) appear or disappear.
        # This prevents bounding-box jitter (distance/position flickering) from constantly
        # triggering the AI models and causing API 429 Rate Limits.
        prev_labels = {label for label, _, _ in prev_sig}
        curr_labels = {label for label, _, _ in curr_sig}
        
        return prev_labels != curr_labels

    @staticmethod
    def _is_complex_environment(enriched: List[Detection]) -> bool:
        if len(enriched) >= 4:
            return True
        near_count = sum(1 for det in enriched if str(det.get("distance", "")).lower() == "near")
        return near_count >= 2

    def _should_escalate_to_cloud(
        self,
        mode: AssistantMode,
        *,
        major_scene_change: bool,
        complex_environment: bool,
        avg_conf: float,
        enriched: List[Detection],
    ) -> bool:
        now = time.time()
        dwell_time = now - getattr(self, "_last_major_change_ts", now)
        is_stable_scene_description = (mode == AssistantMode.SCENE_DESCRIPTION) and (dwell_time >= 4.0)

        if not major_scene_change and not is_stable_scene_description:
            return False

        if is_stable_scene_description:
            return True

        high_detail = len(enriched) >= self.config.cloud_high_detail_object_threshold
        low_conf = avg_conf < self.config.cloud_low_confidence_threshold

        # Local-first policy: cloud only for genuinely complex/high-detail computation.
        if mode == AssistantMode.SCENE_DESCRIPTION:
            return complex_environment or high_detail or low_conf

        if mode in {AssistantMode.NAVIGATION, AssistantMode.OBSTACLE_AWARENESS}:
            return complex_environment and low_conf

        return low_conf and high_detail

    @staticmethod
    def _clock_from_position(position: str) -> str:
        pos = position.strip().lower()
        if pos == "left":
            return "10 o'clock"
        if pos == "right":
            return "2 o'clock"
        return "12 o'clock"

    @staticmethod
    def _steps_from_distance(distance: str) -> str:
        dist = distance.strip().lower()
        if dist == "near":
            return "about one step"
        if dist == "medium":
            return "about three steps"
        return "about five steps"

    def _fallback_instruction(self, mode: AssistantMode, scene_text: str) -> str:
        if mode == AssistantMode.NAVIGATION:
            return "You are doing well. Turn a little toward 11 o'clock and take two short steps."
        if mode == AssistantMode.OBSTACLE_AWARENESS:
            return "Pause here for a second, then scan gently left and right with your cane."
        if "no objects" in scene_text.lower():
            return "It feels open ahead around 12 o'clock. Take two slow, careful steps forward."
        return "I notice nearby objects. Slow down, keep your cane moving, and stay centered."

    def _sanitize_instruction(self, raw_text: str, mode: AssistantMode, scene_text: str) -> str:
        text = " ".join(str(raw_text).strip().split())
        if not text:
            return self._fallback_instruction(mode, scene_text)

        lowered = text.lower()
        banned_meta = [
            "if i want",
            "as an ai",
            "i think",
            "i would",
            "reason",
            "because",
            "let me",
        ]
        if any(token in lowered for token in banned_meta):
            return self._fallback_instruction(mode, scene_text)

        unsafe = [
            "approach the person",
            "approach person",
            "follow the person",
            "walk to the person",
            "touch",
            "run",
        ]
        if any(token in lowered for token in unsafe):
            return self._fallback_instruction(mode, scene_text)

        # Scene description mode: allow rich multi-sentence output from Gemini
        if mode == AssistantMode.SCENE_DESCRIPTION:
            # Just clean special characters and cap length
            cleaned = re.sub(r"[^a-zA-Z0-9,.\-\s!?;:']", "", text)
            cleaned = " ".join(cleaned.split())
            words = cleaned.split()
            if len(words) < 4:
                return self._fallback_instruction(mode, scene_text)
            if any(len(w) > 16 for w in words):
                return self._fallback_instruction(mode, scene_text)
            # Cap at ~75 words to keep TTS vivid but concise.
            if len(words) > 75:
                cleaned = " ".join(words[:75])
            return cleaned

        # Navigation/obstacle modes: keep exactly one short sentence.
        parts = re.split(r"(?<=[.!?])\s+", text)
        sentence = parts[0].strip(" .!?")
        sentence = re.sub(r"[^a-zA-Z0-9,\-\s]", "", sentence)
        sentence = " ".join(sentence.split())

        words = sentence.split()
        if len(words) < 4:
            return self._fallback_instruction(mode, scene_text)
        if len(words) > 24:
            sentence = " ".join(words[:24])

        # Filter likely malformed/hallucinated token bursts.
        if any(len(w) > 16 for w in sentence.split()):
            return self._fallback_instruction(mode, scene_text)

        # Force actionable navigation-safe intent.
        allowed_verbs = {
            "move", "step", "turn", "keep", "stop", "pause", "scan", "continue",
            "shift", "slow", "go", "avoid", "clear", "watch", "ahead", "path",
            "face", "take", "head", "sweep",
        }
        if not any(w.lower().strip(",") in allowed_verbs for w in sentence.split()):
            return self._fallback_instruction(mode, scene_text)

        cleaned = sentence[0].upper() + sentence[1:] if sentence else sentence
        return f"{cleaned}."

    def _obstacle_safety_assessment(self, enriched: List[Detection]) -> Tuple[bool, str]:
        # Immediate hazard: near + center — name the object so user knows what to avoid
        for det in enriched:
            if det.get("distance") == "near" and det.get("position") == "center":
                label = str(det.get("label", "object")).strip()
                clock = self._clock_from_position("center")
                return True, f"Careful, {label} at {clock}, about one step ahead. Please stop now."
        # Immediate hazard: near + side — name the object and give clear action
        for det in enriched:
            if det.get("distance") == "near" and det.get("position") in {"left", "right"}:
                label = str(det.get("label", "object")).strip()
                side = det.get("position")
                opposite = "right" if side == "left" else "left"
                clock = self._clock_from_position(str(side))
                return True, f"{label.capitalize()} is close near {clock}. Shift slightly to your {opposite}."
        # Early warning: medium + center — give extra reaction time
        for det in enriched:
            if det.get("distance") == "medium" and det.get("position") == "center":
                label = str(det.get("label", "something")).strip()
                steps = self._steps_from_distance("medium")
                return True, f"{label.capitalize()} is around 12 o'clock, {steps} ahead. Slow down gently."
        return False, ""

    def _scene_signature(self, enriched: List[Detection]) -> str:
        """Return a stable hash for coarse scene-change gating."""
        compact: List[Tuple[str, str, str, str]] = []
        for det in enriched:
            label = str(det.get("label", "")).strip().lower()
            pos = str(det.get("position", "")).strip().lower()
            dist = str(det.get("distance", "")).strip().lower()
            conf = det.get("confidence", 0.0)
            try:
                conf_bucket = f"{round(float(conf), 1):.1f}"
            except (TypeError, ValueError):
                conf_bucket = "0.0"

            if label:
                compact.append((label, pos, dist, conf_bucket))

        compact.sort()
        payload = "|".join(f"{a}:{b}:{c}:{d}" for a, b, c, d in compact)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _is_cloud_failure_text(self, text: str) -> bool:
        lowered = text.strip().lower()
        if not lowered:
            return True
        # Only match exact failure phrases from our own fallback strings
        failure_phrases = [
            "cloud guidance unavailable",
            "cloud api call failed",
            "failed",
        ]
        return any(lowered == phrase or lowered.startswith(phrase) for phrase in failure_phrases)

    def _cloud_allowed_by_guardrails(
        self,
        mode: AssistantMode,
        recompute: bool,
        scene_signature: str,
        major_scene_change: bool,
        complex_environment: bool,
    ) -> bool:
        now = time.time()

        if self.stop_event.is_set():
            return False

        if now < self._cloud_breaker_until_ts:
            return False

        if (now - self._last_cloud_attempt_ts) < self.config.cloud_min_interval_sec:
            return False

        periodic_due = self._should_use_long_term_cloud(mode)
        prev_sig = self._last_cloud_signature_by_mode.get(mode.value, "")
        scene_changed = prev_sig != scene_signature

        dwell_time = now - getattr(self, "_last_major_change_ts", now)
        is_stable_scene_description = (mode == AssistantMode.SCENE_DESCRIPTION) and (dwell_time >= 4.0)
        
        # Prevent spam: if we already asked the cloud for this exact scene signature, don't ask again just because we are dwelling.
        if not scene_changed:
            is_stable_scene_description = False

        # Cloud only on major scene changes, periodic context, or long stable dwell times.
        if not major_scene_change and not (periodic_due and complex_environment) and not is_stable_scene_description:
            return False

        # If detections were not recomputed and no periodic sync is due, skip cloud.
        if not recompute and not periodic_due and not is_stable_scene_description:
            return False

        if not scene_changed and not periodic_due and not is_stable_scene_description:
            return False

        return True

    def _call_cloud_with_guardrails(
        self,
        mode: AssistantMode,
        scene_text: str,
        reason: str,
        scene_signature: str,
        recompute: bool,
        major_scene_change: bool,
        complex_environment: bool,
        cloud_needed: bool,
    ) -> Optional[str]:
        if not cloud_needed:
            return None

        if not self._cloud_allowed_by_guardrails(
            mode,
            recompute,
            scene_signature,
            major_scene_change,
            complex_environment,
        ):
            return None

        self._last_cloud_attempt_ts = time.time()
        try:
            cloud_text = call_cloud_api(scene_text, reason, mode.value)
            self.logger.info(
                "Cloud returned (%d chars): %s",
                len(cloud_text),
                cloud_text[:120] if cloud_text else "[EMPTY]",
            )
        except Exception as exc:
            self.logger.error("Cloud API call failed: %s", exc)
            cloud_text = "failed"

        if self._is_cloud_failure_text(cloud_text):
            # Only trip the breaker for genuine failures (network/auth), not empty content
            is_hard_failure = cloud_text.strip().lower() in {"failed", "cloud api call failed"}
            if is_hard_failure:
                self._cloud_failures += 1
                if self._cloud_failures >= self.config.cloud_breaker_failure_threshold:
                    self._cloud_breaker_until_ts = time.time() + self.config.cloud_failure_cooldown_sec
                    self.logger.warning(
                        "Cloud circuit opened for %.1fs after %s consecutive failures.",
                        self.config.cloud_failure_cooldown_sec,
                        self._cloud_failures,
                    )
            else:
                self.logger.warning("Cloud returned empty/unusable response for mode=%s — falling back to local.", mode.value)

            # Graceful degradation: reuse last valid cloud response for same scene if present.
            cached_sig = self._last_cloud_signature_by_mode.get(mode.value, "")
            cached_text = self._last_cloud_response_by_mode.get(mode.value, "")
            if cached_text and cached_sig == scene_signature:
                self.logger.info("Reusing cached cloud response for same scene.")
                return self._sanitize_instruction(cached_text, mode, scene_text)
            return None

        self._cloud_failures = 0
        self._last_cloud_response_by_mode[mode.value] = cloud_text
        self._last_cloud_signature_by_mode[mode.value] = scene_signature
        return self._sanitize_instruction(cloud_text, mode, scene_text)

    def _scene_cloud_response(
        self,
        scene_text: str,
        enriched: List[Detection],
        avg_conf: float,
        mode: AssistantMode,
        recompute: bool,
        scene_signature: str,
        major_scene_change: bool,
        complex_environment: bool,
        cloud_needed: bool,
    ) -> Tuple[str, str]:
        _ = (enriched, avg_conf, mode)
        # Scene mode prefers cloud for better accuracy, but with guardrails and local fallback.
        cloud_text = self._call_cloud_with_guardrails(
            mode=mode,
            scene_text=scene_text,
            reason="scene_description_accuracy",
            scene_signature=scene_signature,
            recompute=recompute,
            major_scene_change=major_scene_change,
            complex_environment=complex_environment,
            cloud_needed=cloud_needed,
        )
        if cloud_text:
            return cloud_text, "cloud"

        if not recompute or not major_scene_change:
            # Still provide local description if objects exist in the scene
            if enriched:
                local_text = self._local_fast_response(mode, scene_text, enriched, avg_conf)
                return self._sanitize_instruction(local_text, mode, scene_text), "local_llm"
            return "", "local_llm"

        local_text = self._local_fast_response(mode, scene_text, enriched, avg_conf)
        return self._sanitize_instruction(local_text, mode, scene_text), "local_llm"

    def _local_fast_response(
        self,
        mode: AssistantMode,
        scene_text: str,
        enriched: List[Detection],
        avg_conf: float,
    ) -> str:
        scene_data = {
            "mode": mode.value,
            "scene": scene_text,
            "detections": enriched,
            "average_confidence": round(avg_conf, 4),
            "instruction_style": "one short safe navigation sentence, no reasoning",
        }
        return query_ollama(scene_data)

    def _should_use_long_term_cloud(self, mode: AssistantMode) -> bool:
        now = time.time()
        last = self._last_cloud_context_ts.get(mode.value, 0.0)
        if (now - last) >= self.config.long_term_cloud_interval_sec:
            self._last_cloud_context_ts[mode.value] = now
            return True
        return False

    def _navigation_response(
        self,
        scene_text: str,
        enriched: List[Detection],
        avg_conf: float,
        recompute: bool,
        scene_signature: str,
        major_scene_change: bool,
        complex_environment: bool,
        cloud_needed: bool,
    ) -> Tuple[str, str, str, bool]:
        hazard, hazard_text = self._obstacle_safety_assessment(enriched)
        if hazard:
            return hazard_text, "rule", "high", True

        if self._scan_is_stale() and not self._scan_active:
            self._start_environment_scan("navigation", force=True)
            return "Before moving forward, let us do a quick 360 scan for a safer route.", "rule", "medium", False

        now = time.time()
        if (
            self._route_origin
            and self._route_destination
            and major_scene_change
            and (now - self._last_nav_prompt_ts) > self.config.nav_instruction_interval_sec
        ):
            route_step = get_next_navigation_instruction(self._route_origin, self._route_destination)
            self._last_nav_prompt_ts = now
            if route_step:
                scene_data = {
                    "mode": AssistantMode.NAVIGATION.value,
                    "scene": scene_text,
                    "map_step": route_step,
                    "instruction_style": "one short safe instruction, no reasoning",
                }
                text = query_ollama(scene_data)
                return self._sanitize_instruction(text, AssistantMode.NAVIGATION, scene_text), "local_llm", "medium", False

        # Navigation mode policy: local for immediate response, cloud periodically for long-term context.
        local_text = ""
        if recompute and major_scene_change:
            local_text = self._local_fast_response(
                AssistantMode.NAVIGATION,
                scene_text,
                enriched,
                avg_conf,
            )
            local_text = self._sanitize_instruction(local_text, AssistantMode.NAVIGATION, scene_text)

        cloud_text = self._call_cloud_with_guardrails(
            mode=AssistantMode.NAVIGATION,
            scene_text=scene_text,
            reason="long_term_context",
            scene_signature=scene_signature,
            recompute=recompute,
            major_scene_change=major_scene_change,
            complex_environment=complex_environment,
            cloud_needed=cloud_needed,
        )
        if cloud_text:
            return cloud_text, "cloud", "medium", False

        if local_text:
            return local_text, "local_llm", "medium", False

        return self._navigation_hint_from_memory(), "rule", "medium", False

    def _inference_loop(self) -> None:
        self.logger.info("Inference thread started.")
        frame_index = 0

        while not self.stop_event.is_set():
            self._poll_voice_commands()

            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            frame_index += 1
            if frame_index % self.config.frame_stride != 0:
                self.frame_queue.task_done()
                continue

            with self._state_lock:
                mode = self._mode

            if mode == AssistantMode.PAUSED:
                self.frame_queue.task_done()
                continue

            now = time.time()
            min_interval = 2.0 if mode == AssistantMode.SCENE_DESCRIPTION else self.config.inference_min_interval_sec
            if (now - self._last_inference_step_ts) < min_interval:
                self.frame_queue.task_done()
                continue
            self._last_inference_step_ts = now

            try:
                recompute = self._should_recompute_detections(frame)
                if recompute:
                    detections = self.detector.process_frame(frame)
                    enriched = self._enrich_detections(detections)
                    scene_text = format_scene(
                        detections,
                        self.detector.get_position,
                        self.detector.estimate_distance,
                    )
                    self._cached_detections = detections
                    self._cached_enriched = enriched
                    self._cached_scene_text = scene_text
                    self._last_detect_ts = time.time()
                else:
                    detections = self._cached_detections
                    enriched = self._cached_enriched
                    scene_text = self._cached_scene_text

                avg_conf = compute_confidence(detections)
                scene_signature = self._scene_signature(enriched)
                semantic_scene = self._semantic_scene_signature(enriched)
                major_scene_change = self._is_major_scene_change(self._last_semantic_scene, semantic_scene)
                complex_environment = self._is_complex_environment(enriched)
                cloud_needed = self._should_escalate_to_cloud(
                    mode,
                    major_scene_change=major_scene_change,
                    complex_environment=complex_environment,
                    avg_conf=avg_conf,
                    enriched=enriched,
                )

                if recompute:
                    self._last_semantic_scene = semantic_scene
                    if major_scene_change:
                        self._last_major_change_ts = time.time()

                scan_was_active = self._scan_active
                scan_text = self._update_environment_scan(enriched, recompute)
                scan_in_control = scan_was_active

                response_text = scan_text or ""
                source = "rule"
                priority = "medium" if response_text else "low"
                interrupt = False

                if mode == AssistantMode.OBSTACLE_AWARENESS:
                    hazard, hazard_text = self._obstacle_safety_assessment(enriched)
                    if hazard:
                        response_text = hazard_text
                        source = "rule"
                        priority = "high"
                        interrupt = True
                    else:
                        if not scan_in_control:
                            with self._llm_lock:
                                if not self._is_llm_running:
                                    self._is_llm_running = True
                                    threading.Thread(
                                        target=self._async_llm_worker,
                                        args=(mode, scene_text, enriched, avg_conf, recompute, scene_signature, major_scene_change, complex_environment, cloud_needed),
                                        daemon=True
                                    ).start()

                        now = time.time()
                        if not response_text and (now - self._last_mode_status_ts) >= self.config.mode_status_interval_sec:
                            if scan_in_control:
                                response_text = "Nice and slow. Keep turning while I map each direction."
                            else:
                                response_text = "Area clear so far. Keep moving carefully."
                            source = "rule"
                            priority = "low"
                            self._last_mode_status_ts = now
                else:
                    # For all other modes, dispatch to async LLM only when scan workflow is idle.
                    if not scan_in_control:
                        with self._llm_lock:
                            if not self._is_llm_running:
                                self._is_llm_running = True
                                threading.Thread(
                                    target=self._async_llm_worker,
                                    args=(mode, scene_text, enriched, avg_conf, recompute, scene_signature, major_scene_change, complex_environment, cloud_needed),
                                    daemon=True
                                ).start()

                # Keep user informed even in stable scenes via fast rules
                if not response_text and not self._is_llm_running and (time.time() - self._last_spoken_ts) >= self.config.speech_keepalive_interval_sec:
                    if mode == AssistantMode.SCENE_DESCRIPTION and not major_scene_change:
                        response_text = "I am still with you. No major changes right now."
                        source = "rule"
                        priority = "low"
                    elif mode == AssistantMode.NAVIGATION and not major_scene_change:
                        response_text = "Path looks steady. Take one careful step and keep your cane sweeping."
                        source = "rule"
                        priority = "low"

                self.logger.info("mode=%s detections=%s", mode.value, enriched)
                self.logger.info("confidence=%.3f decision_source=%s recompute=%s", avg_conf, source, recompute)

                if response_text:
                    category = "safety" if priority == "high" else ("navigation" if mode == AssistantMode.NAVIGATION else "scene")
                    queued = speak(response_text, priority=priority, interrupt=interrupt, category=category)
                    self.logger.info("speech_queued=%s priority=%s", queued, priority)
                    if queued:
                        self._last_spoken_ts = time.time()

            except Exception as exc:
                self.logger.exception("Pipeline failure: %s", exc)
                speak("Something went wrong. Please pause and scan around carefully.", priority="high", interrupt=True, category="safety")
            finally:
                self.frame_queue.task_done()

        self.logger.info("Inference thread stopped.")


if __name__ == "__main__":
    system = RealTimeAISystem(RuntimeConfig(frame_stride=3))
    system.start()
