from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import DEFAULT_BLUR_CATEGORIES, DEFAULT_SEGMENTATION_CLASSES


def _normalize_classes(raw: Any) -> list[str]:
    if raw is None:
        return list(DEFAULT_SEGMENTATION_CLASSES)
    if isinstance(raw, dict):
        # Expecting {id: name}; order by numeric key if possible, else alpha
        try:
            return [raw[k] for k in sorted(raw, key=lambda v: int(v))]
        except (ValueError, TypeError):
            return [raw[k] for k in sorted(raw)]
    if isinstance(raw, list | tuple):
        return [str(item) for item in raw]
    return list(DEFAULT_SEGMENTATION_CLASSES)


def _normalize_default_blur(raw: Any, classes: list[str]) -> list[str]:
    if not raw:
        return [c for c in classes if c in DEFAULT_BLUR_CATEGORIES]
    if isinstance(raw, list | tuple):
        normalized = [str(item) for item in raw if str(item)]
    else:
        normalized = [str(raw)]
    # Filter to only known classes to avoid surprises
    class_set = {c.lower(): c for c in classes}
    return [class_set[name.lower()] for name in normalized if name.lower() in class_set]


def load_model_metadata(model_path: Path | str) -> dict[str, Any]:
    """
    Load sidecar metadata (classes + defaults) for a given ONNX model.

    Returns defaults when the sidecar is missing or invalid.
    """
    path = Path(model_path)
    meta_path = path.with_suffix(".classes.json")
    classes = list(DEFAULT_SEGMENTATION_CLASSES)
    default_blur = [c for c in classes if c in DEFAULT_BLUR_CATEGORIES]

    try:
        data = json.loads(meta_path.read_text())
    except FileNotFoundError:
        return {"classes": classes, "default_blur": default_blur}
    except json.JSONDecodeError:
        return {"classes": classes, "default_blur": default_blur}

    classes = _normalize_classes(data.get("classes"))
    default_blur = _normalize_default_blur(data.get("default_blur"), classes)
    if not default_blur:
        default_blur = [c for c in classes if c in DEFAULT_BLUR_CATEGORIES]

    return {
        "classes": classes,
        "default_blur": default_blur,
        "metadata_path": str(meta_path),
    }
