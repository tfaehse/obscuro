"""
Anonymizer core package for dashcam video anonymization.

This package provides functionality for detecting, tracking, and blurring
bikes, heads, people, plates, and vehicles in dashcam videos.
"""

from .config import (
    AnonymizerConfig,
    ConfigLayers,
    apply_overrides,
    create_config_template,
    get_config,
    load_config,
    with_overrides,
)
from .core import Anonymizer

__all__ = [
    "Anonymizer",
    "AnonymizerConfig",
    "ConfigLayers",
    "apply_overrides",
    "create_config_template",
    "get_config",
    "load_config",
    "with_overrides",
]
