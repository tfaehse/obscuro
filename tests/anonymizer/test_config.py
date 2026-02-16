"""
Tests for the anonymizer configuration system.
"""

import tempfile
from pathlib import Path

import pytest
import toml

from anonymizer.config import (
    DEFAULT_TRACKER_PARAMS,
    AnonymizerConfig,
    BlurConfig,
    BlurType,
    ConfigLayers,
    DetectionConfig,
    ModelConfig,
    TrackerParams,
    TrackerType,
    TrackingConfig,
    VideoConfig,
    apply_overrides,
    create_config_template,
    get_config,
    load_config,
    set_config,
    with_overrides,
)
from anonymizer.paths import DEFAULT_MODEL_NAME, get_detection_models_dir


class TestModelConfig:
    """Test ModelConfig class."""

    def test_model_config_defaults(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.name == DEFAULT_MODEL_NAME
        assert config.file is None

    def test_model_path_property(self):
        """Test model path property."""
        config = ModelConfig(name="test_model")
        expected_path = get_detection_models_dir() / "test_model.onnx"
        assert config.path == expected_path

    def test_model_config_with_custom_name(self):
        """Test model configuration with custom name."""
        config = ModelConfig(name="custom_model")
        assert config.name == "custom_model"


class TestBlurConfig:
    """Test BlurConfig class."""

    def test_blur_config_defaults(self):
        """Test default blur configuration."""
        config = BlurConfig()
        assert config.type == BlurType.GAUSSIAN
        assert config.strength == 10

    def test_blur_config_custom_values(self):
        """Test blur configuration with custom values."""
        config = BlurConfig(type=BlurType.PIXELATE, strength=20)
        assert config.type == BlurType.PIXELATE
        assert config.strength == 20

    def test_blur_strength_validation(self):
        """Test blur strength validation."""
        with pytest.raises(ValueError):
            BlurConfig(strength=0)  # Too low
        with pytest.raises(ValueError):
            BlurConfig(strength=101)  # Too high


class TestDetectionConfig:
    """Test DetectionConfig class."""

    def test_detection_config_defaults(self):
        """Test default detection configuration."""
        config = DetectionConfig()
        assert config.confidence_threshold == 0.5
        assert config.low_score_threshold == 0.1
        assert config.inference_size == 1920
        assert config.sahi_overlap_ratio == 0.2
        assert config.single_pass is False

    def test_detection_thresholds_validation(self):
        """Test detection threshold validation."""
        with pytest.raises(ValueError):
            DetectionConfig(confidence_threshold=-0.1)  # Too low
        with pytest.raises(ValueError):
            DetectionConfig(confidence_threshold=1.1)  # Too high


class TestTrackingConfig:
    """Test TrackingConfig class."""

    def test_tracking_config_defaults(self):
        """Test default tracking configuration."""
        config = TrackingConfig()
        assert config.type == TrackerType.BYTETRACK
        assert config.use_offline_linker is True
        params = config.effective_params()
        assert isinstance(params, TrackerParams)
        assert (
            params.distance_gate == DEFAULT_TRACKER_PARAMS[TrackerType.BYTETRACK]["distance_gate"]
        )
        assert params.use_low_score_pool is True
        assert params.offline_linker_max_misses == 30
        assert params.offline_linker_per_frame_gate == 0.05

    def test_params_validation_merges_defaults(self):
        """When overriding params partially, defaults should be preserved."""
        cfg = TrackingConfig(
            type=TrackerType.BYTETRACK,
            params={"max_misses_M": 12},
        )
        assert cfg.use_offline_linker is True
        params = cfg.effective_params()
        assert params.max_misses_M == 12
        assert params.use_low_score_pool is True
        # ensure defaults for ByteTrack applied
        assert params.offline_linker_max_misses == 30
        assert params.offline_linker_per_frame_gate == 0.05

    def test_switching_tracker_type_resets_to_defaults(self):
        """Changing tracker type should discard previous overrides and use new defaults."""
        cfg = TrackingConfig(
            type=TrackerType.BYTETRACK,
            params={"max_misses_M": 12, "distance_gate": 0.25},
        )
        assert cfg.params["max_misses_M"] == 12
        assert cfg.params["distance_gate"] == 0.25

        cfg.type = TrackerType.DUMMY

        dummy_defaults = TrackerParams(**DEFAULT_TRACKER_PARAMS[TrackerType.DUMMY]).model_dump()
        assert cfg.params == dummy_defaults


class TestVideoConfig:
    """Test VideoConfig class."""

    def test_video_config_defaults(self):
        """Test default video configuration."""
        config = VideoConfig()
        assert config.codec == "h264"
        assert config.quality is None

    def test_video_quality_validation(self):
        """Test video quality validation."""
        config = VideoConfig(quality=23)
        assert config.quality == 23

        with pytest.raises(ValueError):
            VideoConfig(quality=0)  # Too low
        with pytest.raises(ValueError):
            VideoConfig(quality=52)  # Too high


class TestAnonymizerConfig:
    """Test main AnonymizerConfig class."""

    def test_anonymizer_config_defaults(self):
        """Test default anonymizer configuration."""
        config = AnonymizerConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.blur, BlurConfig)
        assert isinstance(config.detection, DetectionConfig)
        assert isinstance(config.tracking, TrackingConfig)
        assert isinstance(config.video, VideoConfig)
        assert not config.debug
        assert config.log_level == "INFO"
        assert config.detection.inference_size == 1920
        assert config.detection.sahi_overlap_ratio == 0.2
        assert config.detection.single_pass is False

    def test_anonymizer_config_custom_values(self):
        """Test anonymizer configuration with custom values."""
        config = AnonymizerConfig(
            model=ModelConfig(name="custom_model"),
            blur=BlurConfig(type=BlurType.PIXELATE),
            debug=True,
            log_level="DEBUG",
        )
        assert config.model.name == "custom_model"
        assert config.blur.type == BlurType.PIXELATE
        assert config.debug is True
        assert config.log_level == "DEBUG"

    def test_from_toml_file_not_found(self):
        """Test loading from non-existent TOML file."""
        with pytest.raises(FileNotFoundError):
            AnonymizerConfig.from_toml("non_existent_file.toml")

    def test_from_toml_valid_file(self, mock_config_file):
        """Test loading from valid TOML file."""
        config = AnonymizerConfig.from_toml(mock_config_file)
        assert config.model.name == "test_model"
        assert config.blur.type == BlurType.GAUSSIAN
        assert config.blur.strength == 10


class TestConfigManagement:
    """Test global configuration management."""

    def test_set_and_get_config(self, sample_config):
        """Test setting and getting global configuration."""
        set_config(sample_config)
        retrieved_config = get_config()
        assert retrieved_config.model.name == sample_config.model.name
        assert retrieved_config.blur.type == sample_config.blur.type

    def test_load_config_from_file(self, mock_config_file):
        """Test loading configuration from file."""
        config = load_config(mock_config_file)
        assert config.model.name == "test_model"
        # Test that it's set as global config
        global_config = get_config()
        assert global_config.model.name == "test_model"
        assert global_config.tracking.type == TrackerType.BYTETRACK

    def test_load_config_defaults(self):
        """Test loading configuration without a file returns defaults."""
        config = load_config()
        assert isinstance(config, AnonymizerConfig)
        assert config.tracking.type == TrackerType.BYTETRACK

    def test_load_config_with_overrides(self):
        """Test load_config applies overrides on top of defaults."""
        config = load_config(
            config_path=None,
            overrides={"detection": {"inference_size": 1024}},
            apply=False,
        )
        assert config.detection.inference_size == 1024


class TestConfigTemplate:
    """Test configuration template creation."""

    def test_create_config_template(self, temp_dir):
        """Test creating configuration template file."""
        template_path = temp_dir / "template_config.toml"
        create_config_template(template_path)

        assert template_path.exists()
        content = template_path.read_text()
        assert "model" in content
        assert "blur" in content
        assert "detection" in content
        assert "tracking" in content

        # Test that the template can be loaded as valid TOML
        config_data = toml.load(template_path)
        assert "model" in config_data
        assert "blur" in config_data

        # Test that config can be loaded from template
        config = AnonymizerConfig.from_toml(template_path)
        assert isinstance(config, AnonymizerConfig)


class TestOverrideHelpers:
    """Tests for configuration override utilities."""

    def test_with_overrides_returns_new_instance(self):
        base = AnonymizerConfig()
        updated = with_overrides(base, detection__inference_size=1024)

        assert updated is not base
        assert base.detection.inference_size == 1920
        assert updated.detection.inference_size == 1024

    def test_with_overrides_merges_nested_dicts(self):
        base = AnonymizerConfig()
        updated = with_overrides(
            base,
            {"detection": {"inference_size": 1024}},
            detection__sahi_overlap_ratio=0.3,
        )

        assert updated.detection.inference_size == 1024
        assert updated.detection.sahi_overlap_ratio == 0.3

    def test_apply_overrides_is_shorthand(self):
        base = AnonymizerConfig()
        updated = apply_overrides(base, {"blur": {"strength": 25}})

        assert updated.blur.strength == 25


class TestConfigLayers:
    """Tests for the ConfigLayers helper class."""

    def test_layers_resolve_base_only(self):
        base = AnonymizerConfig()
        layers = ConfigLayers(base)
        resolved = layers.resolve()

        assert resolved == base
        assert resolved is not base  # always return a new instance

    def test_layers_with_named_override(self):
        base = AnonymizerConfig()
        layers = ConfigLayers(base)
        layers.set_layer("cli", {"detection": {"inference_size": 1024}})

        resolved = layers.resolve()
        assert resolved.detection.inference_size == 1024
        assert base.detection.inference_size == 1920

        cumulative = layers.cumulative_overrides()
        assert cumulative == {"detection": {"inference_size": 1024}}

    def test_layers_remove_and_extra_overrides(self):
        base = AnonymizerConfig()
        layers = ConfigLayers(base)
        layers.set_layer("server", {"detection": {"inference_size": 1024}})
        layers.set_layer("cli", {"detection": {"batch_size": 2}})

        resolved = layers.resolve({"detection": {"sahi_overlap_ratio": 0.4}})
        assert resolved.detection.inference_size == 1024
        assert resolved.detection.batch_size == 2
        assert resolved.detection.sahi_overlap_ratio == 0.4

        layers.remove_layer("cli")
        resolved_without_cli = layers.resolve()
        assert resolved_without_cli.detection.batch_size == base.detection.batch_size
        assert resolved_without_cli.detection.inference_size == 1024


class TestBlurType:
    """Test BlurType enum."""

    def test_blur_type_values(self):
        """Test BlurType enum values."""
        assert BlurType.GAUSSIAN == "gaussian"
        assert BlurType.PIXELATE == "pixelate"
        assert BlurType.BLACKOUT == "blackout"
        assert BlurType.DEBUG == "debug"

    def test_blur_type_from_string(self):
        """Test creating BlurType from string."""
        assert BlurType("gaussian") == BlurType.GAUSSIAN
        assert BlurType("pixelate") == BlurType.PIXELATE
        assert BlurType("debug") == BlurType.DEBUG


class TestTrackerType:
    """Test TrackerType enum."""

    def test_tracker_type_values(self):
        """Test TrackerType enum values."""
        assert TrackerType.DUMMY == "dummy"
        assert TrackerType.BYTETRACK == "bytetrack"
        assert TrackerType.BOTSORT == "botsort"
        assert TrackerType.HYBRID_SOT == "hybrid_sot"

    def test_tracker_type_from_string(self):
        """Test creating TrackerType from string."""
        assert TrackerType("dummy") == TrackerType.DUMMY
        assert TrackerType("bytetrack") == TrackerType.BYTETRACK
        assert TrackerType("botsort") == TrackerType.BOTSORT
        assert TrackerType("hybrid_sot") == TrackerType.HYBRID_SOT


class TestConfigTomlSaving:
    """Test configuration TOML file saving functionality."""

    def test_to_toml_method(self):
        """Test saving configuration to TOML file."""
        config = AnonymizerConfig(
            model=ModelConfig(name="custom_model"), blur=BlurConfig(strength=15), debug=True
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "test_config.toml"

            # Save to TOML file
            config.to_toml(config_path)

            # Verify file was created
            assert config_path.exists()

            # Verify content is valid TOML and contains expected values
            saved_data = toml.load(config_path)
            assert saved_data["model"]["name"] == "custom_model"
            assert saved_data["blur"]["strength"] == 15
            assert saved_data["debug"] is True

    def test_to_toml_with_string_path(self):
        """Test saving configuration to TOML file with string path."""
        config = AnonymizerConfig()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = str(Path(tmp_dir) / "test_config.toml")

            # Should work with string path
            config.to_toml(config_path)

            # Verify file was created
            assert Path(config_path).exists()


class TestGetTrackerKwargs:
    """Test the get_tracker_kwargs method."""

    def test_get_tracker_kwargs_returns_tracker_params(self):
        cfg = AnonymizerConfig(
            tracking=TrackingConfig(
                type=TrackerType.BOTSORT,
                params={"distance_gate_lo": 0.25, "cam_motion_comp": True},
            )
        )

        kwargs = cfg.get_tracker_kwargs()
        assert "params" in kwargs
        params = kwargs["params"]
        assert isinstance(params, TrackerParams)
        assert params.distance_gate_lo == 0.25
        assert params.cam_motion_comp is True


class TestGetConfigGlobal:
    """Test the global configuration functions."""

    def test_get_config_returns_instance(self):
        """Test that get_config returns a configuration instance."""
        # Clear any existing global config
        import anonymizer.config

        anonymizer.config._config_instance = None

        config = get_config()

        assert isinstance(config, AnonymizerConfig)

        # Should return the same instance on subsequent calls
        config2 = get_config()
        assert config is config2

    def test_get_config_with_existing_instance(self):
        """Test get_config when global instance already exists."""
        # Set up a specific config instance
        test_config = AnonymizerConfig(debug=True)

        import anonymizer.config

        anonymizer.config._config_instance = test_config

        config = get_config()

        # Should return the existing instance
        assert config is test_config
        assert config.debug is True
