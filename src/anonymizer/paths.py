"""Utilities for locating persistent storage directories."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from importlib import resources
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "obscuro"
APP_AUTHOR = "obscuro"
ENV_DATA_DIR = "BLUR_DATA_DIR"
ENV_MODELS_DIR = "BLUR_MODELS_DIR"

DEFAULT_MODEL_NAMES = ("1280_nano", "640_nano", "1280_nano_b1", "640_nano_b1")
DEFAULT_MODEL_NAME = DEFAULT_MODEL_NAMES[0]
DEFAULT_MODEL_FILENAME = f"{DEFAULT_MODEL_NAME}.onnx"
DEFAULT_MODEL_FILENAMES = {name: f"{name}.onnx" for name in DEFAULT_MODEL_NAMES}
IMMUTABLE_MODEL_NAMES = set(DEFAULT_MODEL_NAMES)
_BUNDLED_MODELS_PACKAGE = "anonymizer._bundled_models"


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


def _get_bundled_model(filename: str):
    try:
        bundled_root = resources.files(_BUNDLED_MODELS_PACKAGE)
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    candidate = bundled_root / filename
    return candidate if candidate.is_file() else None


def _get_repo_model(filename: str) -> Path | None:
    """Fallback: locate a model shipped alongside the source tree."""
    repo_root = Path(__file__).resolve().parents[2]
    models_dir = repo_root / "models"
    if not models_dir.exists():
        return None

    direct_candidate = models_dir / filename
    if direct_candidate.is_file():
        return direct_candidate

    # Grab the first available ONNX model as a safety net.
    candidates = sorted(models_dir.glob("*.onnx"))
    return candidates[0] if candidates else None


def _ensure_model_present(name: str, models_dir: Path) -> Path | None:
    filename = DEFAULT_MODEL_FILENAMES.get(name, f"{name}.onnx")
    target = models_dir / filename
    if target.exists():
        return target

    bundled = _get_bundled_model(filename)
    if bundled is None:
        bundled = _get_repo_model(filename)
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
    models_dir = models_dir or get_models_dir()
    ensured: list[Path] = []
    for name in DEFAULT_MODEL_NAMES:
        path = _ensure_model_present(name, models_dir)
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
    "get_models_dir",
]
