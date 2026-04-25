"""Frame differencing utility for lightweight scene-change detection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class SceneChangeConfig:
    diff_threshold: float = 10.0
    force_interval_seconds: float = 2.0


class SceneChangeDetector:
    def __init__(self, config: Optional[SceneChangeConfig] = None) -> None:
        self.config = config or SceneChangeConfig()
        self._prev_gray: Optional[np.ndarray] = None
        self._last_process_ts = 0.0

    def should_recompute(self, frame: Any) -> bool:
        now = time.time()
        if self._prev_gray is None:
            self._prev_gray = self._to_gray(frame)
            self._last_process_ts = now
            return True

        gray = self._to_gray(frame)
        diff = cv2.absdiff(self._prev_gray, gray)
        mean_diff = float(diff.mean())

        force_due = (now - self._last_process_ts) >= self.config.force_interval_seconds
        changed = mean_diff >= self.config.diff_threshold

        if changed or force_due:
            self._prev_gray = gray
            self._last_process_ts = now
            return True

        return False

    @staticmethod
    def _to_gray(frame: Any) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)
