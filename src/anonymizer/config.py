"""Centralized configuration management for the anonymizer system.

Relies on typed Pydantic models with baked-in defaults and optional TOML overrides.
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import toml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from anonymizer.constants import DEFAULT_BLUR_CATEGORIES
from anonymizer.paths import (
    DEFAULT_MODEL_NAME,
    ensure_default_model_present,
    get_detection_models_dir,
)


class BlurType(str, Enum):
    """Available blur types."""

    GAUSSIAN = "gaussian"
    PIXELATE = "pixelate"
    BLACKOUT = "blackout"
    DEBUG = "debug"


class TrackerType(str, Enum):
    """Supported multi-object trackers."""

    DUMMY = "dummy"
    BYTETRACK = "bytetrack"
    BOTSORT = "botsort"
    HYBRID_SOT = "hybrid_sot"
    FUSED = "fused"
    OC_SORT = "oc_sort"


class ModelConfig(BaseModel):
    """Model-related configuration."""

    name: str | None = Field(
        default=DEFAULT_MODEL_NAME, description="Model name (without .onnx extension)"
    )
    file: Path | None = Field(default=None, description="Full path to an ONNX model file")

    @property
    def path(self) -> Path:
        """Resolve the full path to the configured model file."""
        if self.file:
            return Path(self.file).expanduser()
        if self.name:
            if self.name == DEFAULT_MODEL_NAME:
                ensure_default_model_present()
            return get_detection_models_dir() / f"{self.name}.onnx"
        raise ValueError("Model path is not configured; set model.name or model.file")

    # Note: Model validation temporarily disabled for development
    # Can be enabled once models are available

    model_config = ConfigDict(validate_assignment=True)


class BlurConfig(BaseModel):
    """Blur-related configuration."""

    type: BlurType = Field(default=BlurType.GAUSSIAN, description="Type of blur to apply")
    strength: int = Field(default=10, ge=1, le=100, description="Blur strength/intensity")

    model_config = ConfigDict(validate_assignment=True)


class DetectionConfig(BaseModel):
    """Detection-related configuration."""

    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Global detection confidence threshold"
    )
    low_score_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum confidence retained before NMS (also used by low-score pools)",
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        le=256,
        description="Number of frames/images processed per detector forward pass (higher can improve throughput on GPU)",
    )
    disable_masks: bool = Field(
        default=False,
        description="If true, skip segmentation masks and use bounding boxes only",
    )
    inference_size: int = Field(
        default=1920,
        ge=256,
        le=8192,
        description="Longest image edge (in pixels) used for detection inference",
    )
    sahi_overlap_ratio: float = Field(
        default=0.2,
        ge=0.0,
        lt=1.0,
        description="Fractional overlap for SAHI tiles (0-1)",
    )
    single_pass: bool = Field(
        default=False,
        description="Force SAHI single-tile mode (overrides inference_size to model tile size and overlap to 0)",
    )
    classes_to_blur: list[str] = Field(
        default_factory=lambda: list(DEFAULT_BLUR_CATEGORIES),
        description="Names of detector classes to blur (default matches model metadata)",
    )

    model_config = ConfigDict(validate_assignment=True)


class TrackerParams(BaseModel):
    """Normalized tracker hyper-parameters shared across implementations."""

    distance_gate: float = Field(0.4, ge=0.0, le=1.0)
    confirm_after_N: int = Field(2, ge=1, le=10)
    max_misses_M: int = Field(8, ge=1, le=120)
    offline_linker_max_misses: int = Field(30, ge=1, le=600)
    offline_linker_per_frame_gate: float = Field(0.05, ge=0.0, le=1.0)
    use_low_score_pool: bool = False
    use_visual_tracker: bool = False
    vt_max_age: int = Field(6, ge=0, le=120)
    bbox_dilate_pct: float = Field(0.2, ge=0.0, le=0.6)
    temporal_smooth_alpha: float = Field(0.65, ge=0.0, le=1.0)
    ema_alpha: float = Field(0.6, ge=0.0, le=1.0)
    embedding_similarity_gate: float = Field(0.55, ge=0.0, le=1.0)
    min_detection_rate: float = Field(0.0, ge=0.0, le=1.0)
    distance_gate_hi: float = Field(0.05, ge=0.0, le=1.0)
    distance_gate_lo: float = Field(0.02, ge=0.0, le=1.0)
    cam_motion_comp: bool = False
    flow_backend: str = Field("LK")
    vt_backend: str = Field("TrackerNano")
    drift_gate: float = Field(0.15, ge=0.0, le=2.0)
    process_noise: float = Field(1.0, ge=0.0, le=10.0)


DEFAULT_TRACKER_PARAMS: dict[TrackerType, dict[str, Any]] = {
    TrackerType.BYTETRACK: {
        "distance_gate": 0.05,
        "confirm_after_N": 2,
        "max_misses_M": 10,
        "offline_linker_max_misses": 30,
        "offline_linker_per_frame_gate": 0.05,
        "bbox_dilate_pct": 0.2,
        "temporal_smooth_alpha": 1.0,
        "use_low_score_pool": True,
        "process_noise": 1.0,
    },
    TrackerType.DUMMY: {
        "confirm_after_N": 1,
        "max_misses_M": 1,
        "offline_linker_max_misses": 30,
        "offline_linker_per_frame_gate": 0.05,
        "ema_alpha": 0.6,
        "bbox_dilate_pct": 0.15,
        "temporal_smooth_alpha": 0.7,
    },
    TrackerType.BOTSORT: {
        "confirm_after_N": 3,
        "max_misses_M": 5,
        "offline_linker_max_misses": 30,
        "offline_linker_per_frame_gate": 0.025,
        "bbox_dilate_pct": 0.2,
        "temporal_smooth_alpha": 1.0,
        "use_low_score_pool": True,
        "distance_gate_hi": 0.05,
        "distance_gate_lo": 0.02,
        "cam_motion_comp": True,
        "flow_backend": "LK",
    },
    TrackerType.HYBRID_SOT: {
        "distance_gate": 0.05,
        "confirm_after_N": 5,
        "max_misses_M": 2,
        "offline_linker_max_misses": 30,
        "offline_linker_per_frame_gate": 0.05,
        "bbox_dilate_pct": 0.25,
        "temporal_smooth_alpha": 1.0,
        "use_visual_tracker": True,
        "vt_backend": "TrackerNano",
        "vt_max_age": 10,
        "drift_gate": 0.05,
    },
    TrackerType.FUSED: {
        "distance_gate": 0.1,
        "confirm_after_N": 3,
        "max_misses_M": 5,
        "offline_linker_max_misses": 30,
        "offline_linker_per_frame_gate": 0.05,
        "bbox_dilate_pct": 0.2,
        "temporal_smooth_alpha": 1.0,
        "use_low_score_pool": True,
        "distance_gate_hi": 0.08,
        "distance_gate_lo": 0.15,
    },
    TrackerType.OC_SORT: {
        "distance_gate": 0.05,
        "confirm_after_N": 2,
        "max_misses_M": 10,
        "offline_linker_max_misses": 30,
        "offline_linker_per_frame_gate": 0.05,
        "bbox_dilate_pct": 0.2,
        "temporal_smooth_alpha": 0.9,
        "use_low_score_pool": True,
        "process_noise": 1.0,
    },
}


class TrackingConfig(BaseModel):
    """Tracking-related configuration with explicit defaults per tracker."""

    type: TrackerType = Field(default=TrackerType.BYTETRACK, description="Type of tracker to use")
    params: dict[str, Any] = Field(default_factory=dict, description="Effective tracker parameters")
    use_offline_linker: bool = Field(
        default=True, description="Run an offline linking pass after tracking"
    )

    _overrides: dict[str, Any] = PrivateAttr(default_factory=dict)
    _applied_type: TrackerType | None = PrivateAttr(default=None)

    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="after")
    def _apply_defaults(self) -> TrackingConfig:
        previous_type = self._applied_type
        defaults = TrackerParams(**DEFAULT_TRACKER_PARAMS.get(self.type, {})).model_dump()

        if previous_type is not None and previous_type != self.type:
            # Switching tracker types resets overrides; ignore incoming params payload.
            overrides: dict[str, Any] = {}
            params_input: dict[str, Any] = {}
        else:
            overrides = dict(self._overrides)
            params_input = dict(self.params)

        merged = {**defaults, **overrides, **params_input}
        try:
            validated = TrackerParams(**merged).model_dump()
        except (ValueError, TypeError) as exc:  # pragma: no cover - surfaced to caller
            raise ValueError(f"Invalid tracker params for {self.type.value}: {exc}") from exc

        diff = {key: value for key, value in validated.items() if value != defaults.get(key)}

        object.__setattr__(self, "params", validated)
        object.__setattr__(self, "_overrides", diff)
        object.__setattr__(self, "_applied_type", self.type)
        return self

    def default_params(self) -> TrackerParams:
        """Return tracker defaults for the currently selected tracker."""
        return TrackerParams(**DEFAULT_TRACKER_PARAMS.get(self.type, {}))

    def effective_params(self) -> TrackerParams:
        """Return defaults combined with overrides (mirrors params contents)."""
        return TrackerParams(**self.params)

    def param_overrides(self) -> dict[str, Any]:
        """Expose the current parameter overrides relative to the tracker defaults."""
        return dict(self._overrides)

    def update_params(self, updates: dict[str, Any]) -> None:
        """Apply parameter overrides relative to the active tracker defaults."""
        if not updates:
            return
        defaults = TrackerParams(**DEFAULT_TRACKER_PARAMS.get(self.type, {})).model_dump()
        merged = {**defaults, **self._overrides, **updates}
        validated = TrackerParams(**merged).model_dump()
        diff = {key: value for key, value in validated.items() if value != defaults.get(key)}
        object.__setattr__(self, "params", validated)
        object.__setattr__(self, "_overrides", diff)
        object.__setattr__(self, "_applied_type", self.type)


class VideoConfig(BaseModel):
    """Video processing configuration."""

    codec: str = Field(default="h264", description="Output video codec")
    quality: int | None = Field(
        default=None, ge=1, le=51, description="Video quality (lower = better)"
    )

    model_config = ConfigDict(validate_assignment=True)


class AnonymizerConfig(BaseModel):
    """Main configuration class combining all settings."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    blur: BlurConfig = Field(default_factory=BlurConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)

    # Global settings
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    model_config = ConfigDict(validate_assignment=True)

    @classmethod
    def from_toml(cls, config_path: str | Path) -> AnonymizerConfig:
        """Load configuration from a TOML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file {config_path} not found")

        with open(config_path) as f:
            config_data = toml.load(f)

        return cls(**config_data)

    def to_toml(self, config_path: str | Path) -> None:
        """Save configuration to a TOML file."""
        config_path = Path(config_path)
        config_data = self.model_dump()

        with open(config_path, "w") as f:
            toml.dump(config_data, f)

    def get_tracker_kwargs(self) -> dict[str, Any]:
        """Expose validated tracker parameters for factory creation."""
        return {"params": self.tracking.effective_params()}


# Global configuration instance
_config_instance: AnonymizerConfig | None = None


class ConfigLayers:
    """
    Helper to compose configuration from an ordered set of override layers.

    Each layer is a mapping that can contain nested data or dotted/__ separated keys.
    Layers are applied in insertion order on top of the base configuration.
    """

    def __init__(self, base: AnonymizerConfig | None = None):
        self._base: AnonymizerConfig = base if base is not None else AnonymizerConfig()
        self._layers: OrderedDict[str, Mapping[str, Any]] = OrderedDict()

    @property
    def base(self) -> AnonymizerConfig:
        """Access the base configuration (copy to avoid accidental mutation)."""
        return copy.deepcopy(self._base)

    def set_base(self, base: AnonymizerConfig) -> None:
        """Replace the base layer."""
        self._base = base

    def set_layer(self, name: str, overrides: Mapping[str, Any] | None) -> None:
        """Insert or replace a named override layer (remove when overrides is None/empty)."""
        if overrides:
            self._layers[name] = copy.deepcopy(overrides)
        else:
            self._layers.pop(name, None)

    def remove_layer(self, name: str) -> None:
        """Remove a layer by name if present."""
        self._layers.pop(name, None)

    def get_layer(self, name: str) -> Mapping[str, Any] | None:
        """Return a copy of the requested layer (if any)."""
        overrides = self._layers.get(name)
        return copy.deepcopy(overrides) if overrides is not None else None

    def iter_layers(self) -> Iterable[tuple[str, Mapping[str, Any]]]:
        """Yield (name, overrides) for configured layers in order."""
        for name, overrides in self._layers.items():
            yield name, copy.deepcopy(overrides)

    def resolve(
        self,
        *extra_overrides: Mapping[str, Any] | None,
    ) -> AnonymizerConfig:
        """
        Resolve the layered configuration into a new `AnonymizerConfig`.

        Extra overrides (if provided) are applied last without mutating the stored layers.
        """
        merged = self._base.model_dump()
        for overrides in self._layers.values():
            merged = _deep_merge(merged, _normalize_overrides(overrides))
        for overrides in extra_overrides:
            if overrides:
                merged = _deep_merge(merged, _normalize_overrides(overrides))
        return AnonymizerConfig(**merged)

    def cumulative_overrides(self) -> dict[str, Any]:
        """Return the combined override mapping represented by all layers."""
        combined: dict[str, Any] = {}
        for overrides in self._layers.values():
            combined = _deep_merge(combined, _normalize_overrides(overrides))
        return combined

    def snapshot(self) -> tuple[AnonymizerConfig, OrderedDict[str, Mapping[str, Any]]]:
        """Return copies of the current base and layer definitions."""
        return copy.deepcopy(self._base), OrderedDict(
            (name, copy.deepcopy(layer)) for name, layer in self._layers.items()
        )

    def copy(self) -> ConfigLayers:
        """Clone the layered configuration definition."""
        clone = ConfigLayers(copy.deepcopy(self._base))
        for name, overrides in self._layers.items():
            clone._layers[name] = copy.deepcopy(overrides)
        return clone


def _split_override_key(key: str) -> tuple[str, ...]:
    if "__" in key:
        return tuple(part for part in key.split("__") if part)
    if "." in key:
        return tuple(part for part in key.split(".") if part)
    return (key,)


def _normalize_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        if raw_value is None:
            continue
        if isinstance(raw_value, Mapping):
            child = _normalize_overrides(raw_value)
            if not child:
                continue
            existing = normalized.get(raw_key, {})
            if isinstance(existing, dict):
                normalized[raw_key] = _deep_merge(existing, child)
            else:
                normalized[raw_key] = child
            continue
        parts = _split_override_key(raw_key)
        target = normalized
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Override path '{raw_key}' conflicts with existing value")
        target[parts[-1]] = raw_value
    return normalized


def _deep_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_config() -> AnonymizerConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AnonymizerConfig()
    return _config_instance


def set_config(config: AnonymizerConfig) -> None:
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config


def with_overrides(
    config: AnonymizerConfig,
    *override_maps: Mapping[str, Any],
    **override_kwargs: Any,
) -> AnonymizerConfig:
    """
    Return a new config with the provided overrides applied.

    Supports both nested dict overrides and dotted/double-underscore keys.
    """
    if not override_maps and not override_kwargs:
        return config

    layers = ConfigLayers(config)
    for idx, overrides in enumerate(override_maps):
        if overrides:
            layers.set_layer(f"layer_{idx}", overrides)
    if override_kwargs:
        layers.set_layer("layer_kwargs", override_kwargs)
    return layers.resolve()


def apply_overrides(
    config: AnonymizerConfig,
    overrides: Mapping[str, Any] | None = None,
) -> AnonymizerConfig:
    """Convenience wrapper to apply a single mapping of overrides."""
    return with_overrides(config, overrides or {})


def load_config(
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    *,
    apply: bool = True,
) -> AnonymizerConfig:
    """
    Load configuration from defaults + optional TOML + optional overrides.

    Args:
        config_path: Path to TOML config file. If None, use default values.
        overrides: Mapping or sequence of mappings to apply on top.
        apply: Whether to set the loaded config as global default.

    Returns:
        Configuration instance
    """
    config = AnonymizerConfig.from_toml(config_path) if config_path else AnonymizerConfig()

    if overrides is not None:
        if isinstance(overrides, Mapping):
            config = apply_overrides(config, dict(overrides))
        else:
            config = with_overrides(config, *list(overrides))

    if apply:
        set_config(config)
    return config


# Example configuration file template
CONFIG_TEMPLATE = """# Anonymizer Configuration File
# All values shown are defaults

[model]
# name = "uploaded_model"  # Fill with a managed model name (without .onnx)
# file = "/full/path/to/model.onnx"

[blur]
type = "gaussian"
strength = 10

[detection]
confidence_threshold = 0.5
low_score_threshold = 0.1
batch_size = 8
# inference_size = 1920
# sahi_overlap_ratio = 0.2
# single_pass = true
classes_to_blur = ["plate", "head"]

[tracking]
type = "bytetrack"
use_offline_linker = true

[tracking.params]
# Only include keys here to override the defaults for the selected tracker.
# bbox_dilate_pct = 0.2
# temporal_smooth_alpha = 0.6
# ema_alpha = 0.6
# distance_gate = 0.4
# confirm_after_N = 2
# max_misses_M = 10
# offline_linker_max_misses = 30
# use_low_score_pool = true
# distance_gate_hi = 0.05
# distance_gate_lo = 0.02
# cam_motion_comp = true
# flow_backend = "LK"
# use_visual_tracker = false
# vt_backend = "TrackerNano"
# vt_max_age = 6
# drift_gate = 0.15
# process_noise = 1.0

[video]
codec = "h264"
# quality = 23  # Uncomment and set for custom quality

# Global settings
debug = false
log_level = "INFO"
"""


def create_config_template(path: str | Path) -> None:
    """Create a configuration template file."""
    config_path = Path(path)
    with open(config_path, "w") as f:
        f.write(CONFIG_TEMPLATE.strip())


def model_requires_static_batch(model_name: str | None) -> bool:
    """Return True when the selected model enforces batch size 1."""
    if not model_name:
        return False
    return model_name.endswith("_b1")


def enforce_model_batch_constraints(
    config: AnonymizerConfig,
    *,
    log: logging.Logger | None = None,
) -> bool:
    """
    Ensure the configuration honors model-specific batch size constraints.

    Returns True if the batch size was modified.
    """
    model_name = getattr(config.model, "name", None)
    if model_requires_static_batch(model_name) and config.detection.batch_size != 1:
        if log:
            log.info(
                "Model %s enforces batch size 1 (CoreML export). Overriding configured batch size.",
                model_name,
            )
        config.detection.batch_size = 1
        return True
    return False
