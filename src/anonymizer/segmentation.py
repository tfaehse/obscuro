from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.special import expit


def decode_yolo_masks(
    coeffs: np.ndarray | None,
    proto: np.ndarray | None,
    boxes_xyxy: np.ndarray,
    meta: dict[str, Any],
    imgsz: int,
) -> list[dict[str, Any] | None]:
    if coeffs is None or proto is None or coeffs.size == 0 or boxes_xyxy.size == 0:
        return [None] * len(boxes_xyxy)

    original_h, original_w = map(int, meta.get("original_shape", (0, 0)))
    global_h, global_w = map(int, meta.get("global_shape", (original_h, original_w)))

    if original_h <= 0 or original_w <= 0:
        return [None] * len(boxes_xyxy)

    proto_img = proto
    if proto_img.ndim == 4:
        proto_img = proto_img[0]
    if proto_img.ndim == 2:
        side = int(np.sqrt(proto_img.shape[1]))
        proto_img = proto_img.reshape(proto_img.shape[0], side, side)
    if proto_img.ndim != 3:
        return [None] * len(boxes_xyxy)

    mask_dim = proto_img.shape[0]
    if coeffs.shape[1] != mask_dim:
        return [None] * len(boxes_xyxy)

    proto_img = np.asarray(proto_img, dtype=np.float32)
    coeffs = np.asarray(coeffs, dtype=np.float32)
    proto_flat = proto_img.reshape(mask_dim, -1)
    masks = _sigmoid(coeffs @ proto_flat).reshape(-1, proto_img.shape[1], proto_img.shape[2])

    pad_x, pad_y = meta.get("pad", (0.0, 0.0))
    scale_x, scale_y = meta.get("scale", (1.0, 1.0))
    offset_x, offset_y = meta.get("offset", (0.0, 0.0))
    pad_x = float(pad_x)
    pad_y = float(pad_y)
    offset_x = float(offset_x)
    offset_y = float(offset_y)
    valid_w = max(1, min(imgsz, round(original_w * scale_x)))
    valid_h = max(1, min(imgsz, round(original_h * scale_y)))

    mask_h, mask_w = masks.shape[1], masks.shape[2]
    if imgsz <= 0 or mask_h == 0 or mask_w == 0:
        return [None] * len(boxes_xyxy)

    scale_w = mask_w / float(imgsz)
    scale_h = mask_h / float(imgsz)
    pad_x_idx = int(np.clip(round(pad_x * scale_w), 0, mask_w - 1))
    pad_y_idx = int(np.clip(round(pad_y * scale_h), 0, mask_h - 1))
    valid_w_idx = int(np.clip(round(valid_w * scale_w), 1, mask_w - pad_x_idx))
    valid_h_idx = int(np.clip(round(valid_h * scale_h), 1, mask_h - pad_y_idx))

    cropped = masks[
        :,
        pad_y_idx : pad_y_idx + valid_h_idx,
        pad_x_idx : pad_x_idx + valid_w_idx,
    ]
    if cropped.shape[1] == 0 or cropped.shape[2] == 0:
        return [None] * len(boxes_xyxy)

    cropped_h, cropped_w = cropped.shape[1], cropped.shape[2]
    if cropped_h == 0 or cropped_w == 0:
        return [None] * len(boxes_xyxy)

    # Map from original image coordinates -> cropped mask indices.
    # valid_w/valid_h correspond to the unpadded, letterboxed region that was scaled
    # from the original image; cropped_w/cropped_h are that same region in mask space.
    factor_w = cropped_w / float(max(valid_w, 1))
    factor_h = cropped_h / float(max(valid_h, 1))

    payloads: list[dict[str, Any] | None] = [None] * len(boxes_xyxy)

    for i, (mask_map, box) in enumerate(zip(cropped, boxes_xyxy, strict=False)):
        # Clamp box to original image bounds
        x1_f = float(np.clip(box[0], 0, global_w))
        y1_f = float(np.clip(box[1], 0, global_h))
        x2_f = float(np.clip(box[2], 0, global_w))
        y2_f = float(np.clip(box[3], 0, global_h))
        if x2_f <= x1_f or y2_f <= y1_f:
            continue

        bw = int(np.ceil(x2_f - x1_f))
        bh = int(np.ceil(y2_f - y1_f))
        if bw <= 0 or bh <= 0:
            continue

        # Map bbox corners from original image coords to cropped mask coords.
        # Original coords -> scaled (letterboxed) coords via scale_x/scale_y,
        # then scaled coords -> mask indices via factor_w/factor_h.
        cx1 = int(np.floor(x1_f * scale_x * factor_w))
        cy1 = int(np.floor(y1_f * scale_y * factor_h))
        cx2 = int(np.ceil(x2_f * scale_x * factor_w))
        cy2 = int(np.ceil(y2_f * scale_y * factor_h))

        cx1 = max(0, min(cx1, cropped_w))
        cx2 = max(0, min(cx2, cropped_w))
        cy1 = max(0, min(cy1, cropped_h))
        cy2 = max(0, min(cy2, cropped_h))
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        roi_proto = mask_map[cy1:cy2, cx1:cx2]
        if roi_proto.size == 0 or not np.any(roi_proto):
            continue

        # Resize only the proto-space ROI to the bbox size in original image space.
        roi_resized = cv2.resize(roi_proto, (bw, bh), interpolation=cv2.INTER_LINEAR)
        binary = roi_resized >= 0.5
        if not np.any(binary):
            continue

        # Clip mask to the bbox intersection with the original image to avoid leaking outside.
        x1_int = int(np.floor(x1_f + offset_x))
        y1_int = int(np.floor(y1_f + offset_y))
        x2_int = int(np.ceil(x2_f + offset_x))
        y2_int = int(np.ceil(y2_f + offset_y))

        if global_w > 0 and global_h > 0:
            x1_clip = max(0, min(x1_int, global_w))
            y1_clip = max(0, min(y1_int, global_h))
            x2_clip = max(0, min(x2_int, global_w))
            y2_clip = max(0, min(y2_int, global_h))
        else:
            x1_clip, y1_clip, x2_clip, y2_clip = x1_int, y1_int, x2_int, y2_int

        if x2_clip <= x1_clip or y2_clip <= y1_clip:
            continue

        dx1 = x1_clip - x1_int
        dy1 = y1_clip - y1_int
        dx2 = x2_int - x2_clip
        dy2 = y2_int - y2_clip
        masked = binary
        if dx1 or dy1 or dx2 or dy2:
            masked = masked[
                max(0, dy1) : masked.shape[0] - max(0, dy2),
                max(0, dx1) : masked.shape[1] - max(0, dx2),
            ]
        if masked.size == 0 or not np.any(masked):
            continue

        payloads[i] = {
            "mask": masked.astype(bool, copy=False),
            "x1": x1_clip,
            "y1": y1_clip,
        }

    return payloads


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation function."""
    return expit(np.clip(values, -64.0, 64.0))


__all__ = ["decode_yolo_masks"]
