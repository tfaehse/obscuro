from __future__ import annotations

DEFAULT_SEGMENTATION_CLASSES = [
    "bike",
    "head",
    "person",
    "plate",
    "vehicle",
]

DEFAULT_CATEGORY_MAPPING = {str(idx): name for idx, name in enumerate(DEFAULT_SEGMENTATION_CLASSES)}

DEFAULT_BLUR_CATEGORIES: tuple[str, ...] = ("plate", "head")

__all__ = [
    "DEFAULT_BLUR_CATEGORIES",
    "DEFAULT_CATEGORY_MAPPING",
    "DEFAULT_SEGMENTATION_CLASSES",
]
