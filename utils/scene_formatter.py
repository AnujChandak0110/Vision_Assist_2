"""
Utility to convert structured detections into compact natural language.

Example output:
"person at left, near; chair at center, far"
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

Detection = Dict[str, object]
BBox = Tuple[int, int, int, int]


def format_scene(
    detections: Sequence[Detection],
    get_position: Callable[[BBox], str],
    estimate_distance: Callable[[BBox], str],
) -> str:
    """
    Convert detections into a concise semicolon-separated description.

    Rules:
    - Compact format: "<label> at <position>, <distance>"
    - Avoid repetition by removing exact duplicate phrases
    - Preserve first-seen order
    - Handle empty input
    """
    if not detections:
        return "no objects detected"

    phrases: List[str] = []
    seen = set()

    for det in detections:
        label = str(det.get("label", "object")).strip().lower()
        bbox = det.get("bbox")

        if not label or not isinstance(bbox, tuple) or len(bbox) != 4:
            continue

        position = str(get_position(bbox)).strip().lower()
        distance = str(estimate_distance(bbox)).strip().lower()

        phrase = f"{label} at {position}, {distance}"
        if phrase in seen:
            continue

        seen.add(phrase)
        phrases.append(phrase)

    if not phrases:
        return "no objects detected"

    return "; ".join(phrases)


def format_scene_description(
    detections: Sequence[Detection],
    get_position: Callable[[BBox], str],
    estimate_distance: Callable[[BBox], str],
) -> str:
    """Backward-compatible alias for format_scene."""
    return format_scene(detections, get_position, estimate_distance)


if __name__ == "__main__":
    # Minimal self-test
    sample = [
        {"label": "person", "confidence": 0.93, "bbox": (10, 20, 80, 180)},
        {"label": "chair", "confidence": 0.72, "bbox": (140, 40, 220, 170)},
        {"label": "person", "confidence": 0.91, "bbox": (10, 20, 80, 180)},
    ]

    def mock_get_position(bbox: BBox) -> str:
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2
        if center_x < 100:
            return "left"
        if center_x > 200:
            return "right"
        return "center"

    def mock_estimate_distance(bbox: BBox) -> str:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area > 15000:
            return "near"
        if area > 7000:
            return "medium"
        return "far"

    print(format_scene(sample, mock_get_position, mock_estimate_distance))
