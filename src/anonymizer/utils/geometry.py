from __future__ import annotations

import numpy as np


def tlwh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert a TLWH box to XYXY format."""
    x, y, w, h = box.astype(float)
    return np.array([x, y, x + w, y + h], dtype=float)


def iou_tlwh(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two TLWH boxes."""
    a = tlwh_to_xyxy(box_a)
    b = tlwh_to_xyxy(box_b)
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def nms_tlwh(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Apply Non-Maximum Suppression on TLWH boxes.

    :param boxes: Array of shape (N, 4) in TLWH format.
    :param scores: Array of shape (N,) with confidence scores.
    :param iou_threshold: IoU threshold for suppression.
    :return: Indices of boxes to keep.
    """
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if boxes.size == 0:
        return []

    xyxy = np.column_stack(
        (boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3])
    )
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        area_i = max(0.0, (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1]))
        area_rest = np.maximum(
            0.0, (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        )
        union = area_i + area_rest - inter
        iou = np.where(union > 0.0, inter / union, 0.0)

        rest = rest[iou <= iou_threshold]
        order = rest

    return keep
