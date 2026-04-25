"""
Temporal memory for lightweight object persistence in realtime vision.

Stores per object label:
- first_seen
- last_seen
- count
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

Detection = Dict[str, object]
TIME_THRESHOLD = 3.0
STALE_THRESHOLD = 2.0


@dataclass
class MemoryEntry:
    first_seen: float
    last_seen: float
    count: int


class TemporalMemory:
    """Simple label-level persistence memory for frame-by-frame detections."""

    def __init__(
        self,
        time_threshold: float = TIME_THRESHOLD,
        stale_threshold: float = STALE_THRESHOLD,
    ) -> None:
        self.time_threshold = float(time_threshold)
        self.stale_threshold = float(stale_threshold)
        self._memory: Dict[str, MemoryEntry] = {}

    def update_memory(self, detections: Iterable[Detection]) -> None:
        """
        Update memory from current frame detections.

        Each detection is expected to contain at least a "label" key.
        """
        now = time.time()

        for det in detections:
            label = str(det.get("label", "")).strip().lower()
            if not label:
                continue

            entry = self._memory.get(label)
            if entry is None:
                self._memory[label] = MemoryEntry(
                    first_seen=now,
                    last_seen=now,
                    count=1,
                )
            else:
                entry.last_seen = now
                entry.count += 1

    def check_persistence(self) -> Tuple[bool, Optional[str]]:
        """
        Return (True, label) if any object persists beyond TIME_THRESHOLD.

        Only fresh entries (not stale) are considered for persistence.
        """
        now = time.time()
        persisted_label: Optional[str] = None
        longest_duration = -1.0

        for label, entry in self._memory.items():
            if now - entry.last_seen > self.stale_threshold:
                continue

            duration = now - entry.first_seen
            if duration >= self.time_threshold and duration > longest_duration:
                longest_duration = duration
                persisted_label = label

        if persisted_label is None:
            return False, None
        return True, persisted_label

    def cleanup_memory(self) -> None:
        """Remove stale objects that have not been seen recently."""
        now = time.time()
        stale_labels = [
            label
            for label, entry in self._memory.items()
            if now - entry.last_seen > self.stale_threshold
        ]
        for label in stale_labels:
            del self._memory[label]


if __name__ == "__main__":
    # Minimal sanity check
    tm = TemporalMemory()
    tm.update_memory([{"label": "person"}])
    time.sleep(1)
    tm.update_memory([{"label": "person"}, {"label": "chair"}])
    time.sleep(2.1)

    persisted, label = tm.check_persistence()
    print(persisted, label)

    tm.cleanup_memory()
