from __future__ import annotations

DEFAULT_SEGMENTATION_CLASSES = [
    "person",
    "car",
    "bus",
    "motorcycle",
    "truck",
]

DEFAULT_CATEGORY_MAPPING = {str(idx): name for idx, name in enumerate(DEFAULT_SEGMENTATION_CLASSES)}

DEFAULT_BLUR_CATEGORIES: tuple[str, ...] = tuple(DEFAULT_SEGMENTATION_CLASSES)

__all__ = [
    "DEFAULT_BLUR_CATEGORIES",
    "DEFAULT_CATEGORY_MAPPING",
    "DEFAULT_SEGMENTATION_CLASSES",
]
