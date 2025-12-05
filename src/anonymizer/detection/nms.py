from __future__ import annotations

import numpy as np


def compute_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Compute IoU values between a single box and many boxes (XYXY format)."""
    if others.size == 0:
        return np.empty(0, dtype=np.float32)

    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_self = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_others = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0, others[:, 3] - others[:, 1]
    )
    union = area_self + area_others - inter_area
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter_area / union, 0.0)
    return iou.astype(np.float32)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, list[list[int]]]:
    """
    NMS that tracks which boxes were suppressed for each kept box.
    Returns:
        tuple: (keep_indices, contribution_groups)
    """
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64), []

    def nms_single(indices: np.ndarray) -> tuple[list[int], list[list[int]]]:
        local_boxes = boxes[indices]
        local_scores = scores[indices]
        order = local_scores.argsort()[::-1]
        keep_local: list[int] = []
        contribs: list[list[int]] = []

        while order.size > 0:
            current = order[0]
            global_idx = int(indices[current])
            keep_local.append(global_idx)
            if order.size == 1:
                contribs.append([global_idx])
                break
            rest = order[1:]
            ious = compute_iou(local_boxes[current], local_boxes[rest])
            suppress_mask = ious > iou_threshold
            suppressed_global = indices[rest[suppress_mask]].astype(int).tolist()
            contribs.append([global_idx, *suppressed_global])
            order = rest[~suppress_mask]
        return keep_local, contribs

    keep: list[int] = []
    contributions: list[list[int]] = []
    unique_classes = np.unique(classes)
    for cls in unique_classes:
        cls_indices = np.where(classes == cls)[0]
        cls_keep, cls_contribs = nms_single(cls_indices)
        keep.extend(cls_keep)
        contributions.extend(cls_contribs)

    sorted_pairs = sorted(
        zip(keep, contributions, strict=False), key=lambda pair: scores[pair[0]], reverse=True
    )
    keep_sorted = [pair[0] for pair in sorted_pairs]
    contrib_sorted = [pair[1] for pair in sorted_pairs]
    return np.asarray(keep_sorted, dtype=np.int64), contrib_sorted
