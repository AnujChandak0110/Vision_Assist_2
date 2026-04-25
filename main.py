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

import cv2

from audio.tts import speak
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


@dataclass
class RuntimeConfig:
    camera_index: int = 0
    frame_stride: int = 3
    queue_size: int = 2
    detector_debug: bool = False
    min_recompute_interval_sec: float = 2.0
    nav_instruction_interval_sec: float = 5.0
    long_term_cloud_interval_sec: float = 12.0
    mode_status_interval_sec: float = 8.0
    inference_min_interval_sec: float = 0.35
    cloud_min_interval_sec: float = 4.0
    cloud_failure_cooldown_sec: float = 20.0
    cloud_breaker_failure_threshold: int = 3
    scene_stale_refresh_sec: float = 6.0
    speech_keepalive_interval_sec: float = 12.0
    cloud_low_confidence_threshold: float = 0.45
    cloud_high_detail_object_threshold: int = 5


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
        self._mode = AssistantMode.SCENE_DESCRIPTION

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
        speak(
            "Assistant is ready. You can say start scene description mode, start navigation mode, switch to obstacle awareness, or pause.",
            priority="medium",
            category="system",
        )

        if self.voice_listener.is_available():
            speak("Voice commands are active. You can speak now.", priority="low", category="system")
        else:
            speak(
                "Voice commands are unavailable right now. I can still guide you using camera analysis.",
                priority="medium",
                category="system",
            )

        self.voice_listener.start()
        self._camera_thread.start()
        self._inference_thread.start()

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

            speak(
                f"Navigation mode enabled. Getting a walking route from {origin} to {destination}.",
                priority="medium",
                category="navigation",
            )
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

        if mode == AssistantMode.PAUSED:
            speak("Pausing guidance. Say assistant and a mode command to continue.", priority="medium", category="system")
        else:
            readable = mode.value.replace("_", " ")
            speak(f"Mode selected: {readable}.", priority="medium", category="system")

    def _camera_loop(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            self.logger.error("Unable to open camera index %s", self.config.camera_index)
            speak("Camera unavailable. Please check camera connection.", priority="high", interrupt=True, category="safety")
            self.stop_event.set()
            return

        self.logger.info("Camera thread started.")
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                self.logger.warning("Camera read failed.")
                continue

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        cap.release()
        self.logger.info("Camera thread stopped.")

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
        if prev_sig == curr_sig:
            return False

        # Any object appear/disappear, position change, or near/far change is major.
        prev_labels = {label for label, _, _ in prev_sig}
        curr_labels = {label for label, _, _ in curr_sig}
        if prev_labels != curr_labels:
            return True

        return True

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
        if not major_scene_change:
            return False

        high_detail = len(enriched) >= self.config.cloud_high_detail_object_threshold
        low_conf = avg_conf < self.config.cloud_low_confidence_threshold

        # Local-first policy: cloud only for genuinely complex/high-detail computation.
        if mode == AssistantMode.SCENE_DESCRIPTION:
            return complex_environment or high_detail or low_conf

        if mode in {AssistantMode.NAVIGATION, AssistantMode.OBSTACLE_AWARENESS}:
            return complex_environment and low_conf

        return low_conf and high_detail

    def _fallback_instruction(self, mode: AssistantMode, scene_text: str) -> str:
        if mode == AssistantMode.NAVIGATION:
            return "Move slowly and keep slightly left while scanning ahead."
        if mode == AssistantMode.OBSTACLE_AWARENESS:
            return "Pause and scan around before taking the next step."
        if "no objects" in scene_text.lower():
            return "Path seems clear. Move slowly and scan every two steps."
        return "Move slowly and keep scanning for obstacles ahead."

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

        # Keep exactly one short sentence.
        parts = re.split(r"(?<=[.!?])\s+", text)
        sentence = parts[0].strip(" .!?")
        sentence = re.sub(r"[^a-zA-Z0-9,\-\s]", "", sentence)
        sentence = " ".join(sentence.split())

        words = sentence.split()
        if len(words) < 4:
            return self._fallback_instruction(mode, scene_text)
        if len(words) > 16:
            sentence = " ".join(words[:16])

        # Filter likely malformed/hallucinated token bursts.
        if any(len(w) > 16 for w in sentence.split()):
            return self._fallback_instruction(mode, scene_text)

        # Force actionable navigation-safe intent.
        allowed_verbs = {"move", "step", "turn", "keep", "stop", "pause", "scan", "continue", "shift"}
        if not any(w.lower().strip(",") in allowed_verbs for w in sentence.split()):
            return self._fallback_instruction(mode, scene_text)

        cleaned = sentence[0].upper() + sentence[1:] if sentence else sentence
        return f"{cleaned}."

    def _obstacle_safety_assessment(self, enriched: List[Detection]) -> Tuple[bool, str]:
        for det in enriched:
            if det.get("distance") == "near" and det.get("position") == "center":
                return True, "Careful, there is something right in front of you. Please stop for a moment."
        for det in enriched:
            if det.get("distance") == "near" and det.get("position") in {"left", "right"}:
                side = det.get("position")
                opposite = "right" if side == "left" else "left"
                return True, f"There is an obstacle close on your {side}. Move a little to your {opposite}."
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
        return "unavailable" in lowered or "failed" in lowered

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

        # Cloud only on major scene changes or periodic context for complex scenes.
        if not major_scene_change and not (periodic_due and complex_environment):
            return False

        # If detections were not recomputed and no periodic sync is due, skip cloud.
        if not recompute and not periodic_due:
            return False

        if not scene_changed and not periodic_due:
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
        cloud_text = call_cloud_api(scene_text, reason)

        if self._is_cloud_failure_text(cloud_text):
            self._cloud_failures += 1
            if self._cloud_failures >= self.config.cloud_breaker_failure_threshold:
                self._cloud_breaker_until_ts = time.time() + self.config.cloud_failure_cooldown_sec
                self.logger.warning(
                    "Cloud circuit opened for %.1fs after %s consecutive failures.",
                    self.config.cloud_failure_cooldown_sec,
                    self._cloud_failures,
                )

            # Graceful degradation: reuse last valid cloud response for same scene if present.
            cached_sig = self._last_cloud_signature_by_mode.get(mode.value, "")
            cached_text = self._last_cloud_response_by_mode.get(mode.value, "")
            if cached_text and cached_sig == scene_signature:
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

        return local_text, "local_llm", "medium", False

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
            if (now - self._last_inference_step_ts) < self.config.inference_min_interval_sec:
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

                response_text = ""
                source = "rule"
                priority = "medium"
                interrupt = False

                if mode == AssistantMode.OBSTACLE_AWARENESS:
                    hazard, hazard_text = self._obstacle_safety_assessment(enriched)
                    if hazard:
                        response_text = hazard_text
                        source = "rule"
                        priority = "high"
                        interrupt = True
                    else:
                        if recompute and major_scene_change:
                            response_text = self._local_fast_response(
                                AssistantMode.OBSTACLE_AWARENESS,
                                scene_text,
                                enriched,
                                avg_conf,
                            )
                            response_text = self._sanitize_instruction(
                                response_text,
                                AssistantMode.OBSTACLE_AWARENESS,
                                scene_text,
                            )
                            source = "local_llm"
                            priority = "medium"

                        cloud_text = self._call_cloud_with_guardrails(
                            mode=AssistantMode.OBSTACLE_AWARENESS,
                            scene_text=scene_text,
                            reason="long_term_context",
                            scene_signature=scene_signature,
                            recompute=recompute,
                            major_scene_change=major_scene_change,
                            complex_environment=complex_environment,
                            cloud_needed=cloud_needed,
                        )
                        if cloud_text:
                            response_text = cloud_text
                            source = "cloud"

                        now = time.time()
                        if not response_text and (now - self._last_mode_status_ts) >= self.config.mode_status_interval_sec:
                            response_text = "Obstacle awareness is active. I am monitoring the area around you."
                            source = "local_llm"
                            priority = "low"
                            self._last_mode_status_ts = now
                elif mode == AssistantMode.SCENE_DESCRIPTION:
                    # Scene description mode policy: priority cloud for richer contextual accuracy.
                    response_text, source = self._scene_cloud_response(
                        scene_text,
                        enriched,
                        avg_conf,
                        mode,
                        recompute,
                        scene_signature,
                        major_scene_change,
                        complex_environment,
                        cloud_needed,
                    )
                    priority = "low"
                elif mode == AssistantMode.NAVIGATION:
                    response_text, source, priority, interrupt = self._navigation_response(
                        scene_text,
                        enriched,
                        avg_conf,
                        recompute,
                        scene_signature,
                        major_scene_change,
                        complex_environment,
                        cloud_needed,
                    )
                elif mode in {AssistantMode.OBJECT_FINDER, AssistantMode.TEXT_READING}:
                    if recompute and major_scene_change:
                        scene_data = {
                            "mode": mode.value,
                            "scene": scene_text,
                            "detections": enriched,
                            "instruction_style": "one short safe instruction, no reasoning",
                        }
                        response_text = self._sanitize_instruction(query_ollama(scene_data), mode, scene_text)
                        source = "local_llm"
                        priority = "low"

                if response_text:
                    response_text = self._sanitize_instruction(response_text, mode, scene_text)

                # Keep user informed even in stable scenes, but with long cooldown.
                if not response_text and (time.time() - self._last_spoken_ts) >= self.config.speech_keepalive_interval_sec:
                    if mode == AssistantMode.SCENE_DESCRIPTION and not major_scene_change:
                        response_text = "Scene is stable. Move slowly and keep scanning ahead."
                        source = "rule"
                        priority = "low"
                    elif mode == AssistantMode.NAVIGATION and not major_scene_change:
                        response_text = "Path seems unchanged. Continue slowly and scan ahead."
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
