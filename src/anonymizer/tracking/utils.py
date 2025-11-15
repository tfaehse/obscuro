"""Shared tracking utilities for frame prep, crops, and similarity."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

__all__ = [
    "clamp_bbox",
    "cosine_similarities",
    "crop_patch",
    "prepare_frame_bgr",
    "update_weighted_embedding",
]


def prepare_frame_bgr(frame: np.ndarray | None) -> np.ndarray | None:
    """
    Normalize a decoded frame for OpenCV trackers: uint8, contiguous, 3-channel BGR.

    Handles grayscale/BGRA inputs, clips dtype ranges, and enforces contiguity.
    """
    if frame is None:
        return None
    prepared = frame
    if prepared.ndim == 2:
        prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
    elif prepared.ndim == 3:
        channels = prepared.shape[2]
        if channels == 1:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_BGRA2BGR)
        elif channels > 4:
            prepared = prepared[:, :, :3]

    if prepared.dtype != np.uint8:
        prepared = np.clip(prepared, 0, 255).astype(np.uint8)

    if not prepared.flags["C_CONTIGUOUS"]:
        prepared = np.ascontiguousarray(prepared)

    return prepared


def clamp_bbox(
    bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]
) -> tuple[int, int, int, int] | None:
    """Clamp a bbox to frame bounds; return None if it collapses."""
    x, y, w, h = bbox
    h_max, w_max = frame_shape[:2]
    x = max(0, min(x, w_max))
    y = max(0, min(y, h_max))
    w = max(0, min(w, w_max - x))
    h = max(0, min(h, h_max - y))
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def crop_patch(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Crop a clamped bbox from frame; return None if invalid."""
    clamped = clamp_bbox(bbox, frame.shape)
    if clamped is None:
        return None
    x, y, w, h = clamped
    return frame[y : y + h, x : x + w]


def cosine_similarities(history: Iterable[np.ndarray], current: np.ndarray) -> np.ndarray:
    """Cosine similarity between a set of embeddings and a current embedding."""
    history_list = list(history)
    if not history_list:
        return np.zeros(0, dtype=np.float32)
    stacked = np.stack(history_list, axis=0).astype(np.float32, copy=False)
    current = current.astype(np.float32, copy=False)
    norm_hist = np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-8
    norm_cur = np.linalg.norm(current) + 1e-8
    return (stacked @ current) / (norm_hist[:, 0] * norm_cur)


def update_weighted_embedding(
    rep: np.ndarray | None, emb: np.ndarray, weight: float, alpha: float = 0.5
) -> np.ndarray:
    """Update an embedding representative using a weighted EMA."""
    emb_norm = emb.astype(np.float32) / (np.linalg.norm(emb) + 1e-8)
    if rep is None:
        return emb_norm
    weight = max(weight, 1e-3)
    rep_update = rep * (1 - alpha) + emb_norm * (alpha * weight)
    return rep_update / (np.linalg.norm(rep_update) + 1e-8)
