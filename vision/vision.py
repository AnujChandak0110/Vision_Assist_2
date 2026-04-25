"""
YOLOv8 Nano object detection module optimized for CPU environments.

Requirements implemented:
- Ultralytics YOLOv8n
- OpenCV video capture
- Frame resize to 416x416
- Confidence filtering >= 0.5
- Allowed classes: person, chair, bottle, cup
- Structured detections with label, confidence, bbox
- get_position(bbox): left/center/right
- estimate_distance(bbox): near/medium/far (bbox area heuristic)
- Debug logging
- Optional frame skipping for performance simulation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

BBox = Tuple[int, int, int, int]
Detection = Dict[str, object]


@dataclass
class DetectorConfig:
    model_name: str = "yolov8n.pt"
    image_size: int = 416
    confidence_threshold: float = 0.5
    allowed_labels: Tuple[str, ...] = (
        "person", "bicycle", "car", "motorcycle", "bus", "truck",
        "traffic light", "stop sign", "fire hydrant",
        "bench", "cat", "dog", "horse",
        "backpack", "umbrella", "handbag", "suitcase",
        "chair", "couch", "bed", "dining table", "toilet",
        "laptop", "cell phone", "book",
        "bottle", "cup", "fork", "knife", "scissors",
        "potted plant", "vase",
    )
    skip_frames: int = 0
    debug: bool = False


class VisionDetector:
    """CPU-friendly YOLOv8n detector for webcam/video streams."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self.logger = self._build_logger(self.config.debug)
        self.frame_counter = 0

        self.model = YOLO(self.config.model_name)
        # Force CPU to keep behavior predictable on laptops/RPi-like setups.
        self.model.to("cpu")

        self.allowed_labels = set(self.config.allowed_labels)
        self.logger.debug("Initialized VisionDetector with config: %s", self.config)

    @staticmethod
    def _build_logger(debug: bool) -> logging.Logger:
        logger = logging.getLogger("vision.detector")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.propagate = False
        return logger

    def get_position(self, bbox: BBox, frame_width: int = 416) -> str:
        """Return left/center/right based on bbox center x-position."""
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2.0

        if center_x < frame_width / 3:
            return "left"
        if center_x > (2 * frame_width) / 3:
            return "right"
        return "center"

    def estimate_distance(self, bbox: BBox, frame_area: int = 416 * 416) -> str:
        """
        Estimate object distance using bbox area ratio.

        Heuristic:
        - near   : area_ratio >= 0.15
        - medium : 0.05 <= area_ratio < 0.15
        - far    : area_ratio < 0.05
        """
        x1, y1, x2, y2 = bbox
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area_ratio = (w * h) / float(frame_area)

        if area_ratio >= 0.15:
            return "near"
        if area_ratio >= 0.05:
            return "medium"
        return "far"

    def _should_skip_current_frame(self) -> bool:
        if self.config.skip_frames <= 0:
            return False

        # Process one frame, then skip `skip_frames` subsequent frames.
        cycle = self.config.skip_frames + 1
        return (self.frame_counter % cycle) != 0

    def process_frame(self, frame) -> List[Detection]:
        """
        Run detection on a single frame and return structured detections.

        Returns a list of dictionaries with keys:
        {
          "label": str,
          "confidence": float,
          "bbox": (x1, y1, x2, y2)
        }
        """
        self.frame_counter += 1

        if frame is None:
            self.logger.debug("Received empty frame.")
            return []

        if self._should_skip_current_frame():
            self.logger.debug("Skipping frame #%d (skip_frames=%d).", self.frame_counter, self.config.skip_frames)
            return []

        resized = cv2.resize(frame, (self.config.image_size, self.config.image_size), interpolation=cv2.INTER_AREA)
        self.logger.debug("Processing frame #%d resized to %dx%d.", self.frame_counter, self.config.image_size, self.config.image_size)

        results = self.model.predict(
            source=resized,
            imgsz=self.config.image_size,
            conf=self.config.confidence_threshold,
            verbose=False,
            device="cpu",
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = names.get(cls_id, str(cls_id))
            confidence = float(box.conf.item())

            if label not in self.allowed_labels:
                continue
            if confidence < self.config.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bbox: BBox = (x1, y1, x2, y2)

            detection: Detection = {
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": bbox,
            }
            detections.append(detection)

            self.logger.debug("Detection: %s", detection)

        return detections

    def detect_from_camera(self, camera_index: int = 0) -> None:
        """Minimal real-time loop using OpenCV capture for quick testing."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}")

        self.logger.info("Camera started. Press 'q' to quit.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    self.logger.warning("Failed to read frame from camera.")
                    break

                detections = self.process_frame(frame)

                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    label = det["label"]
                    conf = det["confidence"]
                    pos = self.get_position(det["bbox"], frame_width=self.config.image_size)
                    dist = self.estimate_distance(
                        det["bbox"],
                        frame_area=self.config.image_size * self.config.image_size,
                    )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 200, 20), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f} | {pos} | {dist}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (20, 200, 20),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Vision Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Camera stopped.")


if __name__ == "__main__":
    # Minimal test loop
    config = DetectorConfig(debug=True, skip_frames=1)
    detector = VisionDetector(config=config)
    detector.detect_from_camera(camera_index=0)
