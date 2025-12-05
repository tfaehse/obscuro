"""Utilities for locating persistent storage directories."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "obscuro"
APP_AUTHOR = "obscuro"
ENV_DATA_DIR = "BLUR_DATA_DIR"
ENV_MODELS_DIR = "BLUR_MODELS_DIR"
ENV_BUNDLED_MODELS_DIR = "BLUR_BUNDLED_MODELS_DIR"

DEFAULT_MODEL_NAMES = ("1280_nano_seg", "640_nano_seg", "1280_nano_seg_b1", "640_nano_seg_b1")
DEFAULT_MODEL_NAME = DEFAULT_MODEL_NAMES[0]
DEFAULT_MODEL_FILENAME = f"{DEFAULT_MODEL_NAME}.onnx"
DEFAULT_MODEL_FILENAMES = {name: f"{name}.onnx" for name in DEFAULT_MODEL_NAMES}
IMMUTABLE_MODEL_NAMES = set(DEFAULT_MODEL_NAMES)
DETECTION_MODELS_SUBDIR = "detection"
TRACKING_MODELS_SUBDIR = "tracking"


@lru_cache(maxsize=1)
def _default_data_root() -> Path:
    if override := os.environ.get(ENV_DATA_DIR):
        return Path(override).expanduser()

    return Path(user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR, roaming=False))


def get_data_root(create: bool = True) -> Path:
    """Return the root directory for persisted application data."""
    path = _default_data_root()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def _default_models_dir() -> Path:
    if override := os.environ.get(ENV_MODELS_DIR):
        return Path(override).expanduser()
    return get_data_root(create=False) / "models"


def get_models_dir(create: bool = True) -> Path:
    """Return the directory used to persist ONNX models."""
    path = _default_models_dir()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _subdir(base: Path, subdir: str, create: bool) -> Path:
    path = base / subdir
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_detection_models_dir(create: bool = True) -> Path:
    """Return the directory containing detection ONNX models."""
    return _subdir(get_models_dir(create=create), DETECTION_MODELS_SUBDIR, create)


def get_tracking_models_dir(create: bool = True) -> Path:
    """Return the directory containing tracking ONNX models."""
    return _subdir(get_models_dir(create=create), TRACKING_MODELS_SUBDIR, create)


def _iter_bundled_search_dirs(subdir: str | None = None):
    """Yield candidate directories containing bundled models."""
    override = os.environ.get(ENV_BUNDLED_MODELS_DIR)
    if override:
        override_path = Path(override).expanduser()
        if subdir:
            yield override_path / subdir
        yield override_path

    repo_root = Path(__file__).resolve().parents[2] / "models"
    if subdir:
        yield repo_root / subdir
    yield repo_root
    yield Path.cwd() / "models"


def _get_repo_model(filename: str, *, subdir: str | None = None) -> Path | None:
    """Fallback: locate a model shipped alongside the source tree."""
    search_dirs = list(_iter_bundled_search_dirs(subdir=subdir))

    for directory in search_dirs:
        if not directory.exists():
            continue
        candidate = directory / filename
        if candidate.is_file():
            return candidate

    for directory in search_dirs:
        if not directory.exists():
            continue
        candidates = sorted(directory.glob("*.onnx"))
        if candidates:
            return candidates[0]
    return None


def _ensure_model_present(name: str, models_dir: Path, *, subdir: str | None = None) -> Path | None:
    filename = DEFAULT_MODEL_FILENAMES.get(name, f"{name}.onnx")
    target = models_dir / filename
    if target.exists():
        return target

    bundled = _get_repo_model(filename, subdir=subdir)
    if bundled is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with bundled.open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    except OSError:
        return None
    return target


def ensure_required_models_present(models_dir: Path | None = None) -> list[Path]:
    """Ensure all bundled models exist in the persistent models directory."""
    models_dir = models_dir or get_detection_models_dir()
    ensured: list[Path] = []
    for name in DEFAULT_MODEL_NAMES:
        path = _ensure_model_present(name, models_dir, subdir=DETECTION_MODELS_SUBDIR)
        if path:
            ensured.append(path)
    return ensured


def ensure_default_model_present(models_dir: Path | None = None) -> Path | None:
    """Ensure the primary bundled model exists in the persistent models directory."""
    ensured = ensure_required_models_present(models_dir)
    return ensured[0] if ensured else None


__all__ = [
    "DEFAULT_MODEL_FILENAME",
    "DEFAULT_MODEL_FILENAMES",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_NAMES",
    "IMMUTABLE_MODEL_NAMES",
    "ensure_default_model_present",
    "ensure_required_models_present",
    "get_data_root",
    "get_detection_models_dir",
    "get_models_dir",
    "get_tracking_models_dir",
]
