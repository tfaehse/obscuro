from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import onnxruntime as ort

from anonymizer.paths import ensure_required_models_present, get_detection_models_dir

DEFAULT_MODELS_DIR = get_detection_models_dir()
logger = logging.getLogger("obscuro.detection.model")

ExecutionProvider = str | tuple[str, dict[str, str]]


class ModelLoader:
    """Handles loading of ONNX models and execution provider configuration."""

    def __init__(self, model_path: Path | str, execution_providers: list[str] | None = None):
        self.model_path = Path(model_path)
        logger.warning(f"Using model: {self.model_path}")
        self.requested_execution_providers, self.session_opts = self._get_execution_providers(
            execution_providers
        )
        self.session = self._load_model(
            self.model_path, self.requested_execution_providers, self.session_opts
        )
        self.active_execution_providers = self._resolve_active_providers()
        self.execution_provider = (
            self.active_execution_providers[0] if self.active_execution_providers else "unknown"
        )
        self.input_name = self._get_input_name()

    def _get_execution_providers(
        self, forced_providers: list[str] | None = None
    ) -> tuple[list[ExecutionProvider], Any]:
        """
        Get execution providers for ONNX Runtime.

        :param forced_providers: If provided, use these providers directly.
        :return: Tuple of (providers list, session options)
        """
        so = ort.SessionOptions() if hasattr(ort, "SessionOptions") else None

        if forced_providers:
            logger.info(f"Using forced execution providers: {forced_providers}")
            return forced_providers, so

        providers = ort.get_available_providers() if hasattr(ort, "get_available_providers") else []
        if "CUDAExecutionProvider" in providers:
            logger.info("Using CUDAExecutionProvider")
            providers = ["CUDAExecutionProvider"]
        elif "CoreMLExecutionProvider" in providers:
            logger.info("Using CoreMLExecutionProvider")
            providers = [
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "CPUAndNeuralEngine",
                        "RequireStaticInputShapes": "1",
                        "EnableOnSubgraphs": "0",
                    },
                )
            ]
        elif "MPSExecutionProvider" in providers:
            logger.info("Using MPSExecutionProvider")
            providers = ["MPSExecutionProvider"]
        elif "MLComputeExecutionProvider" in providers:
            logger.info("Using MLComputeExecutionProvider")
            providers = ["MLComputeExecutionProvider"]
        elif "DmlExecutionProvider" in providers:
            logger.info("Using DmlExecutionProvider")
            providers = ["DmlExecutionProvider"]
        elif "TensorrtExecutionProvider" in providers:
            logger.info("Using TensorrtExecutionProvider")
            providers = ["TensorrtExecutionProvider"]
        elif "OpenVINOExecutionProvider" in providers:
            logger.info("Using OpenVINOExecutionProvider")
            providers = ["OpenVINOExecutionProvider"]
        elif "QNNExecutionProvider" in providers:
            logger.info("Using QNNExecutionProvider")
            providers = ["QNNExecutionProvider"]
        else:
            logger.info("Using CPUExecutionProvider")
            providers = ["CPUExecutionProvider"]
        return providers, so

    def _load_model(
        self, model_path: Path | str, providers: list[ExecutionProvider], session_opts: Any
    ) -> Any:
        path = self._resolve_model_path(model_path)
        with contextlib.suppress(Exception):
            ort.preload_dlls()
        try:
            return ort.InferenceSession(str(path), providers=providers, sess_options=session_opts)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load ONNX model {path}: {exc}") from exc

    def _resolve_model_path(self, model_path: Path | str) -> Path:
        path = Path(model_path)
        candidates: Iterable[Path]
        candidates = (path,) if path.is_absolute() else (path, DEFAULT_MODELS_DIR / path)

        def try_candidates() -> Path | None:
            for candidate in candidates:
                if candidate.exists():
                    resolved = candidate.resolve()
                    logger.debug("Resolved detector model path to %s", resolved)
                    self.model_path = resolved
                    return resolved
            return None

        resolved = try_candidates()
        if resolved:
            return resolved

        # Populate bundled models into the default directory and retry.
        ensure_required_models_present()
        resolved = try_candidates()
        if resolved:
            return resolved

        search_paths = ", ".join(str(p.resolve()) for p in candidates)
        hint = (
            "Model file not found. Checked: "
            f"{search_paths}. Upload a model via the API, place an ONNX checkpoint under "
            f"{DEFAULT_MODELS_DIR.resolve()}, or update your configuration to reference the file."
        )
        raise FileNotFoundError(hint)

    def _resolve_active_providers(self) -> list[str]:
        raw_providers: list[str] = []
        if hasattr(self.session, "get_providers") and callable(self.session.get_providers):
            try:
                providers = self.session.get_providers()
            except Exception:  # pragma: no cover - defensive
                providers = None
            if isinstance(providers, list | tuple):
                raw_providers = [str(p) for p in providers]
        if not raw_providers:
            raw_providers = [str(p) for p in self.requested_execution_providers]
        return raw_providers

    def _get_input_name(self) -> str:
        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError("Loaded ONNX model does not expose any inputs")
        return inputs[0].name

    def get_status(self) -> dict[str, Any]:
        primary_requested = (
            self.requested_execution_providers[0] if self.requested_execution_providers else None
        )
        primary_active = self.execution_provider if self.execution_provider else None
        status_code = (
            0 if primary_requested == primary_active and primary_active is not None else -1
        )
        return {
            "requested": list(self.requested_execution_providers),
            "active": list(self.active_execution_providers),
            "primary": primary_active,
            "status_code": status_code,
        }
