"""
Lightweight frame change detection for compute optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class FrameChangeConfig:
    diff_threshold: float = 12.0
    min_change_ratio: float = 0.015
    blur_kernel: int = 5


class FrameChangeDetector:
    """Detects significant scene change using grayscale frame differencing."""

    def __init__(self, config: Optional[FrameChangeConfig] = None) -> None:
        self.config = config or FrameChangeConfig()
        self._previous_gray = None

    def has_significant_change(self, frame) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.config.blur_kernel, self.config.blur_kernel), 0)

        if self._previous_gray is None:
            self._previous_gray = gray
            return True

        diff = cv2.absdiff(self._previous_gray, gray)
        _, thresh = cv2.threshold(diff, self.config.diff_threshold, 255, cv2.THRESH_BINARY)

        changed_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)
        self._previous_gray = gray
        return changed_ratio >= self.config.min_change_ratio
