"""
Efficient blur implementations for CPU processing.
"""

import cv2
import numpy as np


def blur_rois(
    img: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    blur_type: str = "gaussian",
    blur_strength: int = 25,
) -> np.ndarray:
    """
    Apply blur to regions of interest using CPU processing.

    Args:
        img: Input image as numpy array (BGR format)
        rois: List of ROIs as (x, y, w, h) tuples in pixel coordinates
        blur_type: Type of blur ("gaussian", "pixelate", "blackout")
        blur_strength: Blur intensity

    Returns:
        Blurred image as numpy array
    """
    result = img.copy()

    for x, y, w, h in rois:
        # Ensure ROI is within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)

        if w <= 0 or h <= 0:
            continue

        roi = result[y : y + h, x : x + w]

        if blur_type == "gaussian":
            # Ensure kernel size is odd
            kernel_size = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
            blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            result[y : y + h, x : x + w] = blurred_roi

        elif blur_type == "pixelate":
            # Pixelation effect
            roi_h, roi_w = roi.shape[:2]
            if roi_h > 0 and roi_w > 0:
                pixel_size = max(1, blur_strength)
                small_h = max(1, roi_h // pixel_size)
                small_w = max(1, roi_w // pixel_size)
                small_roi = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                pixelated_roi = cv2.resize(
                    small_roi, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
                )
                result[y : y + h, x : x + w] = pixelated_roi

        elif blur_type == "blackout":
            # Black rectangle
            result[y : y + h, x : x + w] = 0

    return result


def convert_relative_to_absolute_rois(
    rois: list[tuple[float, float, float, float]], img_shape: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    """
    Convert relative ROI coordinates (0-1) to absolute pixel coordinates.

    Args:
        rois: List of ROIs as (x1, y1, x2, y2) tuples in relative coordinates
        img_shape: Image shape as (height, width)

    Returns:
        List of ROIs as (x, y, w, h) tuples in pixel coordinates
    """
    h, w = img_shape[:2]
    h = max(1, h)
    w = max(1, w)

    absolute_rois = []
    for x1, y1, x2, y2 in rois:
        # Clamp relative coordinates to [0, 1]
        x1 = max(0.0, min(x1, 1.0))
        y1 = max(0.0, min(y1, 1.0))
        x2 = max(0.0, min(x2, 1.0))
        y2 = max(0.0, min(y2, 1.0))

        # Convert to pixel coordinates
        x1_px = round(x1 * w)
        y1_px = round(y1 * h)
        x2_px = round(x2 * w)
        y2_px = round(y2 * h)

        # Ensure within bounds and convert to (x, y, w, h)
        x1_px = max(0, min(x1_px, w))
        y1_px = max(0, min(y1_px, h))
        x2_px = max(x1_px, min(x2_px, w))
        y2_px = max(y1_px, min(y2_px, h))

        width = x2_px - x1_px
        height = y2_px - y1_px

        if width > 0 and height > 0:
            absolute_rois.append((x1_px, y1_px, width, height))

    return absolute_rois
