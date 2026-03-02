"""Realtime prediction smoothing helpers."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class PredictionStabilizer:
    window_size: int = 6
    min_count: int = 6

    def __post_init__(self) -> None:
        self._window: Deque[str | None] = deque(maxlen=self.window_size)

    def update(self, prediction: str | None) -> str | None:
        self._window.append(prediction)
        if len(self._window) < self.window_size:
            return None

        non_empty = [item for item in self._window if item is not None]
        if not non_empty:
            return None

        label, count = Counter(non_empty).most_common(1)[0]
        return label if count >= self.min_count else None

    def reset(self) -> None:
        self._window.clear()
