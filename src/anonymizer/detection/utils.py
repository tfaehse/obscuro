from __future__ import annotations

import cv2
import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -64.0, 64.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def letterbox(
    image: np.ndarray,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if shape[0] == 0 or shape[1] == 0:
        raise ValueError("Cannot process empty image")

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, left = 0, 0
    bottom, right = int(dh), int(dw)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    scale = (
        float(new_unpad[0] / max(shape[1], 1)),
        float(new_unpad[1] / max(shape[0], 1)),
    )
    pad = (float(left), float(top))

    return image, scale, pad


def ensure_channel_last(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] <= 4:
        return np.transpose(image, (1, 2, 0))
    return image


def preprocess_image(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, dict]:
    original_shape = image.shape[:2]
    image, scale, pad = letterbox(image, new_shape=(imgsz, imgsz), auto=False)
    image = image.transpose((2, 0, 1))
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32) / 255.0

    if image.ndim == 3:
        image = image[None]

    meta = {
        "scale": scale,
        "pad": pad,
        "original_shape": original_shape,
    }

    return image, meta


def apply_scaling_and_padding(
    coords: np.ndarray,
    scale: tuple[float, float],
    pad: tuple[float, float],
    original_shape: tuple[int, int],
) -> np.ndarray:
    """
    Apply scaling and padding to coordinates to map them back to original image space.
    coords: (N, 4) array of [x1, y1, x2, y2]
    """
    scale_x, scale_y = scale
    pad_x, pad_y = pad
    original_h, original_w = original_shape

    # Clone to avoid modifying original
    coords = coords.copy()

    # Remove padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y

    # Undo scaling
    coords[:, [0, 2]] /= scale_x
    coords[:, [1, 3]] /= scale_y

    # Clip to image boundaries
    coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0.0, original_w)
    coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0.0, original_h)

    return coords
