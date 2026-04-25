"""
QA test suite for the real-time AI pipeline.

Coverage:
1. Object detection correctness
2. Temporal persistence triggering
3. Decision switching (rule vs local vs cloud)
4. API failure handling
5. No-object scenario

Includes expected outputs, debug strategies, and lightweight performance checks.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
from requests import RequestException
from requests.exceptions import Timeout

# Allow direct execution: python tests/test_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision.decision_engine import decide_action, rule_based_response
from llm.cloud_api import call_cloud_api
from llm.ollama_local import query_ollama
from memory.temporal_memory import TemporalMemory
from utils.scene_formatter import format_scene


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeCoord(list):
    def tolist(self) -> list:
        return list(self)


class _FakeBox:
    def __init__(self, cls_id: int, conf: float, xyxy: list[float]) -> None:
        self.cls = _FakeScalar(cls_id)
        self.conf = _FakeScalar(conf)
        self.xyxy = [_FakeCoord(xyxy)]


class _FakeResult:
    def __init__(self, names: dict[int, str], boxes: List[_FakeBox]) -> None:
        self.names = names
        self.boxes = boxes


class _FakeYOLOModel:
    def __init__(self) -> None:
        self._result = _FakeResult(
            names={0: "person", 1: "dog", 2: "chair", 3: "cup"},
            boxes=[
                _FakeBox(0, 0.92, [10, 20, 100, 180]),   # keep
                _FakeBox(1, 0.99, [40, 50, 140, 200]),   # filtered (label dog)
                _FakeBox(2, 0.49, [60, 70, 160, 240]),   # filtered (conf < 0.5)
                _FakeBox(3, 0.74, [80, 90, 180, 260]),   # keep
            ],
        )

    def to(self, _device: str) -> "_FakeYOLOModel":
        return self

    def predict(self, **_kwargs):
        return [self._result]


class _FakeYOLOFactory:
    def __init__(self, _model_name: str) -> None:
        self._model = _FakeYOLOModel()

    def to(self, device: str):
        return self._model.to(device)

    def predict(self, **kwargs):
        return self._model.predict(**kwargs)


@pytest.mark.unit
def test_object_detection_correctness_with_filtering(monkeypatch):
    """
    Expected output:
    - Only allowed labels with confidence >= 0.5 are returned.
    - Bounding boxes are integer tuples (x1, y1, x2, y2).

    Debug strategy:
    - If failing, print raw model outputs and verify class-name mapping.
    - Confirm confidence threshold and allowed_labels in DetectorConfig.
    """
    import vision.vision as vision_module

    monkeypatch.setattr(vision_module, "YOLO", _FakeYOLOFactory)

    detector = vision_module.VisionDetector(
        vision_module.DetectorConfig(confidence_threshold=0.5, debug=False)
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.process_frame(frame)

    labels = [d["label"] for d in detections]
    assert labels == ["person", "cup"], f"Unexpected filtered labels: {labels}"

    for det in detections:
        bbox = det["bbox"]
        assert isinstance(bbox, tuple) and len(bbox) == 4
        assert all(isinstance(v, int) for v in bbox)
        assert det["confidence"] >= 0.5


@pytest.mark.unit
def test_temporal_persistence_triggering_and_cleanup():
    """
    Expected output:
    - check_persistence() returns (True, 'person') after TIME_THRESHOLD.
    - After stale timeout + cleanup, persistence no longer triggers.

    Debug strategy:
    - Reduce thresholds further if timing jitter occurs on busy systems.
    - Inspect internal memory entries (first_seen/last_seen/count) during failures.
    """
    memory = TemporalMemory(time_threshold=0.05, stale_threshold=0.15)

    memory.update_memory([{"label": "person"}])
    time.sleep(0.06)

    persisted, label = memory.check_persistence()
    assert persisted is True
    assert label == "person"

    time.sleep(0.16)
    memory.cleanup_memory()

    persisted_after_cleanup, _ = memory.check_persistence()
    assert persisted_after_cleanup is False


@pytest.mark.unit
def test_decision_switching_rule_local_cloud():
    """
    Expected output:
    - High confidence -> source 'rule'
    - Mid confidence -> source 'local_llm'
    - Low confidence -> source 'cloud'
    - Persistence flag True -> source 'cloud' regardless of confidence

    Debug strategy:
    - Log avg confidence before decide_action.
    - Verify HIGH_CONF and LOW_CONF constants if boundary cases fail.
    """
    high = [{"label": "person", "confidence": 0.9, "position": "center", "distance": "far"}]
    mid = [{"label": "person", "confidence": 0.6, "position": "left", "distance": "medium"}]
    low = [{"label": "person", "confidence": 0.3, "position": "right", "distance": "near"}]

    _, src_high = decide_action(high, persistence_flag=False)
    _, src_mid = decide_action(mid, persistence_flag=False)
    _, src_low = decide_action(low, persistence_flag=False)
    _, src_persist = decide_action(high, persistence_flag=True)

    assert src_high == "rule"
    assert src_mid == "local_llm"
    assert src_low == "cloud"
    assert src_persist == "cloud"


@pytest.mark.unit
def test_api_failure_handling_for_ollama_and_cloud(monkeypatch):
    """
    Expected output:
    - Ollama timeout returns safe fallback text.
    - Cloud request failure returns safe fallback text.

    Debug strategy:
    - Capture exception paths in logs to verify timeout vs request-error branches.
    - Ensure env vars are loaded before cloud call path is executed.
    """
    import llm.ollama_local as ollama_module
    import llm.cloud_api as cloud_module

    def raise_timeout(*_args, **_kwargs):
        raise Timeout("simulated timeout")

    def raise_request_error(*_args, **_kwargs):
        raise RequestException("simulated cloud error")

    monkeypatch.setattr(ollama_module.requests, "post", raise_timeout)

    monkeypatch.setenv("CLOUD_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key")
    monkeypatch.setattr(cloud_module.requests, "post", raise_request_error)

    ollama_text = query_ollama({"detections": []})
    cloud_text = call_cloud_api("no objects detected", "low_confidence")

    assert "Stop" in ollama_text or "scan" in ollama_text
    assert "Cloud guidance unavailable" in cloud_text


@pytest.mark.unit
def test_no_object_scenario_and_rule_safety_text():
    """
    Expected output:
    - Scene formatter returns 'no objects detected' for empty input.
    - Rule engine returns a cautious movement instruction for empty detections.

    Debug strategy:
    - Validate upstream detection list shape before formatting.
    - Confirm empty path does not attempt cloud/local calls.
    """

    def _pos(_bbox):
        return "center"

    def _dist(_bbox):
        return "far"

    scene = format_scene([], _pos, _dist)
    rule_text = rule_based_response([])

    assert scene == "no objects detected"
    assert "Move slowly" in rule_text or "scan" in rule_text.lower()


@pytest.mark.performance
def test_performance_scene_formatting_and_decision_latency():
    """
    Performance checks (laptop simulation):
    - Scene formatting should remain lightweight for moderate batch sizes.
    - Decision routing should execute quickly.

    Debug strategy:
    - If this regresses, profile string joins and per-detection transformations.
    - Monitor end-to-end loop timing in main thread logs.
    """
    detections = [
        {"label": "person", "confidence": 0.8, "bbox": (10, 20, 100, 150), "position": "left", "distance": "near"}
        for _ in range(1000)
    ]

    start = time.perf_counter()
    _ = format_scene(
        detections,
        lambda _bbox: "left",
        lambda _bbox: "near",
    )
    _ = decide_action(detections, persistence_flag=False)
    elapsed = time.perf_counter() - start

    # Generous bound for CI/laptop variability while still catching regressions.
    assert elapsed < 0.5, f"Performance regression detected: elapsed={elapsed:.4f}s"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
