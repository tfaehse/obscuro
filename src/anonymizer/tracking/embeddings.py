"""Lightweight shared visual embedding helper for trackers."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from anonymizer.paths import get_tracking_models_dir

logger = logging.getLogger("obscuro.tracking.embeddings")

_MODEL_CACHE: dict[Path, EmbeddingModel] = {}
_FALLBACK_MODEL_NAMES = ("mobilenetv3", "mobile_net_v3", "mobilenet", "mbv3")


def _find_default_model() -> Path | None:
    """Locate a reasonable default embedding ONNX in tracking dirs or bundled models."""
    search_dirs: list[Path] = []
    with contextlib.suppress(Exception):
        search_dirs.append(get_tracking_models_dir(create=True))
    repo_models = Path(__file__).resolve().parents[3] / "models"
    search_dirs.append(repo_models / "tracking")
    search_dirs.append(repo_models)

    seen: set[Path] = set()
    for directory in search_dirs:
        if not directory or directory in seen or not directory.exists():
            continue
        seen.add(directory)
        candidates = sorted(directory.glob("*.onnx"))
        if not candidates:
            continue
        for cand in candidates:
            stem = cand.stem.lower()
            if any(key in stem for key in _FALLBACK_MODEL_NAMES):
                return cand
        return candidates[0]
    return None


class EmbeddingModel:
    """Thin wrapper around an ONNX feature extractor."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        # Default to square if dynamic or missing spatial dims
        self.input_h = int(shape[2]) if len(shape) > 2 and shape[2] not in (None, "None") else 224
        self.input_w = int(shape[3]) if len(shape) > 3 and shape[3] not in (None, "None") else 224
        self._run = self.session.run

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        """Compute an embedding for an RGB image patch."""
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("Empty image for embedding")
        resized = cv2.resize(
            image_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
        )
        arr = resized.astype(np.float32) / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        batch = np.expand_dims(chw, 0)
        outputs = self._run(None, {self.input_name: batch})
        emb = outputs[0]
        if emb.ndim > 2:
            emb = np.reshape(emb, (emb.shape[0], -1))
        return emb[0].astype(np.float32)


def get_embedding_model(path: Path | None = None) -> EmbeddingModel | None:
    """Return a cached embedding model instance (or None if unavailable)."""
    model_path = Path(path) if path else _find_default_model()
    if model_path is None:
        logger.warning("No embedding model found")
        return None
    model_path = model_path.resolve()
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    try:
        model = EmbeddingModel(model_path)
        _MODEL_CACHE[model_path] = model
        logger.info("Loaded embedding model from %s", model_path)
        return model
    except Exception:
        logger.exception("Failed to load embedding model from %s", model_path)
        return None


__all__ = ["EmbeddingModel", "get_embedding_model"]
