from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class TrackState(str, Enum):
    """Lifecycle state for a tracked trajectory."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    HIDDEN = "hidden"
    DEAD = "dead"


@dataclass(slots=True)
class Detection:
    """Lightweight detection container (tlwh format in absolute pixels)."""

    frame_idx: int
    tlwh: np.ndarray  # [x, y, w, h]
    score: float
    frame_size: tuple[int, int] | None = None
    is_confident: bool = True
    mask: dict[str, Any] | None = None

    def as_xyah(self) -> np.ndarray:
        return tlwh_to_xyah(self.tlwh)


@dataclass(slots=True)
class TrackObservation:
    """Output row returned to downstream consumers."""

    frame: int
    track_id: int
    tlwh: np.ndarray
    state: TrackState
    age: int
    last_seen: int
    score: float
    should_blur: bool
    frame_size: tuple[int, int] | None = None
    debug_color: tuple[int, int, int] | None = None
    mask: dict[str, Any] | None = None

    def as_dict(self, *, include_debug: bool = False) -> dict[str, Any]:
        x, y, w, h = self.tlwh.tolist()
        data = {
            "frame": self.frame,
            "track_id": self.track_id,
            "x1": float(x),
            "y1": float(y),
            "x2": float(x + w),
            "y2": float(y + h),
            "width": float(w),
            "height": float(h),
            "state": self.state.value,
            "age": int(self.age),
            "last_seen": int(self.last_seen),
            "score": float(self.score),
            "should_blur": bool(self.should_blur),
            "mask": self.mask,
            **(
                {
                    "frame_width": int(self.frame_size[0]),
                    "frame_height": int(self.frame_size[1]),
                }
                if self.frame_size
                else {}
            ),
        }
        if include_debug and self.debug_color is not None:
            data["debug_color"] = tuple(self.debug_color)
        return data


def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
    x, y, w, h = tlwh.astype(float)
    cx = x + w / 2.0
    cy = y + h / 2.0
    a = w / max(h, 1e-6)
    return np.array([cx, cy, a, h], dtype=float)


def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
    cx, cy, a, h = xyah.astype(float)
    w = a * h
    x = cx - w / 2.0
    y = cy - h / 2.0
    return np.array([x, y, w, h], dtype=float)


def normalized_center(tlwh: np.ndarray, frame_size: tuple[int, int] | None) -> np.ndarray | None:
    if frame_size is None:
        return None
    width, height = frame_size
    if width <= 0 or height <= 0:
        return None
    x, y, w, h = tlwh.astype(float)
    cx = (x + w / 2.0) / float(width)
    cy = (y + h / 2.0) / float(height)
    return np.array([cx, cy], dtype=float)


def center_distance(
    tlwh_a: np.ndarray,
    size_a: tuple[int, int] | None,
    tlwh_b: np.ndarray,
    size_b: tuple[int, int] | None,
) -> float:
    center_a = normalized_center(tlwh_a, size_a)
    center_b = normalized_center(tlwh_b, size_b)
    if center_a is None or center_b is None:
        return float("inf")
    return float(np.linalg.norm(center_a - center_b))


def batched_center_distance(
    boxes_a: Iterable[np.ndarray],
    sizes_a: Iterable[tuple[int, int] | None],
    boxes_b: Iterable[np.ndarray],
    sizes_b: Iterable[tuple[int, int] | None],
) -> np.ndarray:
    list_a = list(boxes_a)
    list_sizes_a = list(sizes_a)
    list_b = list(boxes_b)
    list_sizes_b = list(sizes_b)

    if not list_a or not list_b:
        return np.full((len(list_a), len(list_b)), float("inf"), dtype=float)

    centers_a = [
        normalized_center(box, size) for box, size in zip(list_a, list_sizes_a, strict=False)
    ]
    centers_b = [
        normalized_center(box, size) for box, size in zip(list_b, list_sizes_b, strict=False)
    ]

    distances = np.full((len(list_a), len(list_b)), float("inf"), dtype=float)
    for i, ca in enumerate(centers_a):
        if ca is None:
            continue
        for j, cb in enumerate(centers_b):
            if cb is None:
                continue
            distances[i, j] = float(np.linalg.norm(ca - cb))
    return distances
