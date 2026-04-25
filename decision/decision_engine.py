"""
Hybrid AI decision engine.

Decision priority (safety first, minimize cloud usage):
1) If persistence flag is True -> cloud
2) If avg confidence >= HIGH_CONF -> rule-based
3) If avg confidence < LOW_CONF -> cloud
4) Otherwise -> local LLM
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

Detection = Dict[str, object]

HIGH_CONF = 0.75
LOW_CONF = 0.5


def compute_confidence(detections: Iterable[Detection]) -> float:
    """Return average confidence across detections, or 0.0 if none."""
    values: List[float] = []
    for det in detections:
        conf = det.get("confidence", 0.0)
        try:
            values.append(float(conf))
        except (TypeError, ValueError):
            continue

    if not values:
        return 0.0
    return sum(values) / len(values)


def rule_based_response(detections: Iterable[Detection]) -> str:
    """
    Generate a short safety-first rule response from detections.

    Preference order:
    - Near person/object in center -> stop/wait
    - Near obstacle left/right -> steer opposite
    - Otherwise -> proceed slowly and scan
    """
    # Normalize once for lightweight processing.
    normalized: List[Tuple[str, str, str]] = []
    for det in detections:
        label = str(det.get("label", "")).strip().lower()
        pos = str(det.get("position", "")).strip().lower()
        dist = str(det.get("distance", "")).strip().lower()
        if label:
            normalized.append((label, pos, dist))

    if not normalized:
        return "No clear obstacle detected. Move slowly forward and scan continuously."

    for _, pos, dist in normalized:
        if dist == "near" and pos == "center":
            return "Obstacle ahead at center. Stop and wait, then scan again before moving."

    for _, pos, dist in normalized:
        if dist == "near" and pos == "left":
            return "Obstacle near on left. Shift slightly right and move slowly."
        if dist == "near" and pos == "right":
            return "Obstacle near on right. Shift slightly left and move slowly."

    return "Path is not fully clear. Proceed slowly and keep scanning surroundings."


def decide_action(
    detections: Iterable[Detection],
    persistence_flag: bool,
) -> Tuple[str, str]:
    """
    Decide response source and return (response_text, source).

    Source is one of: "rule", "local_llm", "cloud"
    """
    det_list = list(detections)
    avg_conf = compute_confidence(det_list)

    # Safety override: persistent objects are escalated to cloud reasoning.
    if persistence_flag:
        return (
            "Persistent obstacle detected. Requesting cloud-level guidance for safer navigation.",
            "cloud",
        )

    if avg_conf >= HIGH_CONF:
        return rule_based_response(det_list), "rule"

    if avg_conf < LOW_CONF:
        return (
            "Vision confidence is low. Requesting cloud assistance before next movement.",
            "cloud",
        )

    return (
        "Scene is moderately clear. Using local reasoning for the next safe movement step.",
        "local_llm",
    )


if __name__ == "__main__":
    sample = [
        {"label": "person", "confidence": 0.92, "position": "center", "distance": "near"},
        {"label": "chair", "confidence": 0.81, "position": "left", "distance": "medium"},
    ]

    text, source = decide_action(sample, persistence_flag=False)
    print(source, "->", text)
