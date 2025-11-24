"""
Tests for the CLI functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from anonymizer.config import AnonymizerConfig
from blur_cli.cli import (
    blur_image,
    blur_video,
    create_config,
    get_config_for_args,
    list_models,
    main,
)


class TestCLI:
    """Test CLI main functionality."""

    @patch("sys.argv", ["blur-cli", "--help"])
    def test_help_argument(self):
        """Test help argument functionality."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        # Help should exit with code 0
        assert exc_info.value.code == 0

    def test_no_arguments_shows_help(self):
        """Test CLI without arguments shows help and returns 1."""
        with patch("sys.argv", ["blur-cli"]):
            result = main()
            # Should return 1 when no command is provided
            assert result == 1

    @patch("sys.argv", ["blur-cli", "invalid-command"])
    def test_invalid_command_shows_help(self):
        """Test invalid command shows help and returns 1."""
        with pytest.raises(SystemExit) as exc_info:
            main()
        # Invalid command should exit with code 2
        assert exc_info.value.code == 2

    @patch("sys.argv", ["blur-cli", "image", "test.jpg"])
    @patch("blur_cli.cli.blur_image")
    def test_image_command_calls_blur_image(self, mock_blur_image):
        """Test image command calls blur_image function."""
        mock_blur_image.return_value = 0
        result = main()
        assert result == 0
        mock_blur_image.assert_called_once()

    @patch("sys.argv", ["blur-cli", "video", "test.mp4"])
    @patch("blur_cli.cli.blur_video")
    def test_video_command_calls_blur_video(self, mock_blur_video):
        """Test video command calls blur_video function."""
        mock_blur_video.return_value = 0
        result = main()
        assert result == 0
        mock_blur_video.assert_called_once()

    @patch("sys.argv", ["blur-cli", "config"])
    @patch("blur_cli.cli.create_config")
    def test_config_command_calls_create_config(self, mock_create_config):
        """Test config command calls create_config function."""
        mock_create_config.return_value = 0
        result = main()
        assert result == 0
        mock_create_config.assert_called_once()

    @patch("sys.argv", ["blur-cli", "models"])
    @patch("blur_cli.cli.list_models")
    def test_models_command_calls_list_models(self, mock_list_models):
        """Test models command calls list_models function."""
        mock_list_models.return_value = 0
        result = main()
        assert result == 0
        mock_list_models.assert_called_once()

    @patch("sys.argv", ["blur-cli", "--log-level", "DEBUG", "image", "test.jpg"])
    @patch("blur_cli.cli.setup_logging")
    @patch("blur_cli.cli.blur_image")
    def test_log_level_argument(self, mock_blur_image, mock_setup_logging):
        """Test log level argument is passed to setup_logging."""
        mock_blur_image.return_value = 0
        mock_setup_logging.return_value = Mock()
        main()
        mock_setup_logging.assert_called_once_with(
            log_level="DEBUG", json_log_file=None, enable_colors=True
        )

    @patch("sys.argv", ["blur-cli", "--json-log", "test.log", "image", "test.jpg"])
    @patch("blur_cli.cli.setup_logging")
    @patch("blur_cli.cli.blur_image")
    def test_json_log_argument(self, mock_blur_image, mock_setup_logging):
        """Test json log argument is passed to setup_logging."""
        mock_blur_image.return_value = 0
        mock_setup_logging.return_value = Mock()
        main()
        mock_setup_logging.assert_called_once_with(
            log_level="INFO", json_log_file=Path("test.log"), enable_colors=True
        )

    @patch("sys.argv", ["blur-cli", "--no-colors", "image", "test.jpg"])
    @patch("blur_cli.cli.setup_logging")
    @patch("blur_cli.cli.blur_image")
    def test_no_colors_argument(self, mock_blur_image, mock_setup_logging):
        """Test no-colors argument disables colors."""
        mock_blur_image.return_value = 0
        mock_setup_logging.return_value = Mock()
        main()
        mock_setup_logging.assert_called_once_with(
            log_level="INFO", json_log_file=None, enable_colors=False
        )


class TestBlurImage:
    """Test blur_image function."""

    def test_blur_image_nonexistent_file(self):
        """Test blur_image with non-existent input file."""
        args = Mock()
        args.input = "/nonexistent/file.jpg"

        result = blur_image(args)
        assert result == 1

    @patch("blur_cli.cli.Path.exists", return_value=True)
    @patch("blur_cli.cli.Path.is_file", return_value=False)
    def test_blur_image_not_a_file(self, mock_is_file, mock_exists):
        """Test blur_image with input that is not a file."""
        args = Mock()
        args.input = "/some/directory"

        result = blur_image(args)
        assert result == 1

    @patch("blur_cli.cli.Path.exists", return_value=True)
    @patch("blur_cli.cli.Path.is_file", return_value=True)
    @patch("cv2.imread", return_value=None)
    @patch("blur_cli.cli.get_config_for_args")
    def test_blur_image_cannot_load_image(
        self, mock_get_config, mock_imread, mock_is_file, mock_exists
    ):
        """Test blur_image when cv2 cannot load the image."""
        args = Mock()
        args.input = "/path/to/test.jpg"
        args.output = None

        result = blur_image(args)
        assert result == 1

    @patch("blur_cli.cli.Path.exists", return_value=True)
    @patch("blur_cli.cli.Path.is_file", return_value=True)
    @patch("cv2.imread")
    @patch("cv2.imwrite", return_value=False)
    @patch("blur_cli.cli.get_config_for_args")
    @patch("blur_cli.cli.Anonymizer")
    def test_blur_image_cannot_save_result(
        self,
        mock_anonymizer_class,
        mock_get_config,
        mock_imwrite,
        mock_imread,
        mock_is_file,
        mock_exists,
    ):
        """Test blur_image when cv2 cannot save the result."""
        args = Mock()
        args.input = "/path/to/test.jpg"
        args.output = None

        mock_imread.return_value = [[1, 2, 3]]  # Mock image array
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        mock_anonymizer = Mock()
        mock_anonymizer.blur_image_array.return_value = [[4, 5, 6]]
        mock_anonymizer_class.return_value = mock_anonymizer

        result = blur_image(args)
        assert result == 1

    @patch("blur_cli.cli.Path.exists", return_value=True)
    @patch("blur_cli.cli.Path.is_file", return_value=True)
    @patch("cv2.imread")
    @patch("cv2.imwrite", return_value=True)
    @patch("blur_cli.cli.get_config_for_args")
    @patch("blur_cli.cli.Anonymizer")
    def test_blur_image_success(
        self,
        mock_anonymizer_class,
        mock_get_config,
        mock_imwrite,
        mock_imread,
        mock_is_file,
        mock_exists,
    ):
        """Test successful blur_image execution."""
        args = Mock()
        args.input = "/path/to/test.jpg"
        args.output = "/path/to/output.jpg"

        # Mock image as numpy array with shape attribute
        mock_image = Mock()
        mock_image.shape = [480, 640, 3]  # height, width, channels
        mock_imread.return_value = mock_image

        mock_config = Mock()
        mock_get_config.return_value = mock_config

        mock_anonymizer = Mock()
        mock_anonymizer.blur_image_array.return_value = [[4, 5, 6]]
        mock_anonymizer_class.return_value = mock_anonymizer

        result = blur_image(args)
        assert result == 0
        mock_anonymizer_class.assert_called_once_with(
            config=mock_config,
            progress_callback=mock_anonymizer_class.call_args[1]["progress_callback"],
        )

    @patch("blur_cli.cli.Path.exists", return_value=True)
    @patch("blur_cli.cli.Path.is_file", return_value=True)
    @patch("cv2.imread")
    @patch("cv2.imwrite", return_value=True)
    @patch("blur_cli.cli.get_config_for_args")
    @patch("blur_cli.cli.Anonymizer")
    def test_blur_image_auto_output_filename(
        self,
        mock_anonymizer_class,
        mock_get_config,
        mock_imwrite,
        mock_imread,
        mock_is_file,
        mock_exists,
    ):
        """Test blur_image with auto-generated output filename."""
        args = Mock()
        args.input = "/path/to/test.jpg"
        args.output = None  # Should auto-generate

        # Mock image as numpy array with shape attribute
        mock_image = Mock()
        mock_image.shape = [480, 640, 3]  # height, width, channels
        mock_imread.return_value = mock_image

        mock_config = Mock()
        mock_get_config.return_value = mock_config

        mock_anonymizer = Mock()
        mock_anonymizer.blur_image_array.return_value = [[4, 5, 6]]
        mock_anonymizer_class.return_value = mock_anonymizer

        result = blur_image(args)
        assert result == 0

    @patch("blur_cli.cli.Path.exists", return_value=True)
    @patch("blur_cli.cli.Path.is_file", return_value=True)
    @patch("cv2.imread", side_effect=Exception("Test error"))
    @patch("blur_cli.cli.get_config_for_args")
    def test_blur_image_exception_handling(
        self, mock_get_config, mock_imread, mock_is_file, mock_exists
    ):
        """Test blur_image exception handling."""
        args = Mock()
        args.input = "/path/to/test.jpg"
        args.output = None

        result = blur_image(args)
        assert result == 1


class TestBlurVideo:
    """Test blur_video function."""

    def test_blur_video_nonexistent_file(self):
        """Test blur_video with non-existent input file."""
        args = Mock()
        args.input = "/nonexistent/file.mp4"

        result = blur_video(args)
        assert result == 1

    @patch("blur_cli.cli.Path.exists", return_value=True)
    @patch("blur_cli.cli.Path.is_file", return_value=False)
    def test_blur_video_not_a_file(self, mock_is_file, mock_exists):
        """Test blur_video with input that is not a file."""
        args = Mock()
        args.input = "/some/directory"

        result = blur_video(args)
        assert result == 1

    @patch("blur_cli.cli.Path.exists", return_value=True)
    @patch("blur_cli.cli.Path.is_file", return_value=True)
    @patch("blur_cli.cli.get_config_for_args")
    @patch("blur_cli.cli.Anonymizer")
    def test_blur_video_success(
        self, mock_anonymizer_class, mock_get_config, mock_is_file, mock_exists
    ):
        """Test successful blur_video execution."""
        args = Mock()
        args.input = "/path/to/test.mp4"
        args.output = "/path/to/output.mp4"

        mock_config = Mock()
        mock_get_config.return_value = mock_config

        mock_anonymizer = Mock()
        mock_anonymizer.blur_video.return_value = 0
        mock_anonymizer_class.return_value = mock_anonymizer

        result = blur_video(args)
        assert result == 0
        mock_anonymizer.blur_video.assert_called_once_with(Path(args.input), Path(args.output))

    @patch("blur_cli.cli.Path.exists", return_value=True)
    @patch("blur_cli.cli.Path.is_file", return_value=True)
    @patch("blur_cli.cli.get_config_for_args")
    @patch("blur_cli.cli.Anonymizer")
    def test_blur_video_auto_output_filename(
        self, mock_anonymizer_class, mock_get_config, mock_is_file, mock_exists
    ):
        """Test blur_video with auto-generated output filename."""
        args = Mock()
        args.input = "/path/to/test.mp4"
        args.output = None  # Should auto-generate

        mock_config = Mock()
        mock_get_config.return_value = mock_config

        mock_anonymizer = Mock()
        mock_anonymizer.blur_video.return_value = 0
        mock_anonymizer_class.return_value = mock_anonymizer

        result = blur_video(args)
        assert result == 0

    @patch("blur_cli.cli.Path.exists", return_value=True)
    @patch("blur_cli.cli.Path.is_file", return_value=True)
    @patch("blur_cli.cli.get_config_for_args", side_effect=Exception("Test error"))
    def test_blur_video_exception_handling(self, mock_get_config, mock_is_file, mock_exists):
        """Test blur_video exception handling."""
        args = Mock()
        args.input = "/path/to/test.mp4"
        args.output = None

        result = blur_video(args)
        assert result == 1


class TestGetConfigForArgs:
    """Test get_config_for_args function."""

    def test_get_config_no_config_file(self):
        """Test get_config_for_args without config file."""
        args = Mock()
        args.config = None
        args.model = None
        args.embedding_similarity_gate = None
        args.blur_type = None
        args.blur_strength = None
        args.confidence_threshold = None
        args.low_score_threshold = None
        args.tracker_params = None
        args.tracker = None
        args.use_sahi = None
        args.sahi_overlap = None
        args.inference_size = None
        args.offline_linker = None
        args.batch_size = None
        args.video_codec = None
        args.video_quality = None
        args.config_debug = None
        args.log_level = "INFO"

        result = get_config_for_args(args)
        assert isinstance(result, AnonymizerConfig)

    @patch("blur_cli.cli.set_config")
    @patch("blur_cli.cli.load_config")
    def test_get_config_with_config_file(self, mock_load_config, mock_set_config):
        """Test get_config_for_args with config file."""
        args = Mock()
        args.config = "/path/to/config.toml"
        args.model = None
        args.embedding_similarity_gate = None
        args.blur_type = None
        args.blur_strength = None
        args.confidence_threshold = None
        args.low_score_threshold = None
        args.tracker_params = None
        args.tracker = None
        args.use_sahi = None
        args.sahi_overlap = None
        args.inference_size = None
        args.offline_linker = None
        args.batch_size = None
        args.video_codec = None
        args.video_quality = None
        args.config_debug = None
        args.log_level = "INFO"

        mock_config = AnonymizerConfig()
        mock_load_config.return_value = mock_config

        result = get_config_for_args(args)
        assert isinstance(result, AnonymizerConfig)
        assert result.log_level == "INFO"
        assert result is not mock_config
        mock_load_config.assert_called_once_with(
            config_path="/path/to/config.toml",
            overrides=None,
            apply=False,
        )
        mock_set_config.assert_called_once()

    @patch("blur_cli.cli.set_config")
    def test_get_config_apply_overrides(self, mock_set_config):
        """Test get_config_for_args applies CLI overrides."""
        args = Mock()
        args.config = None
        args.model = "test_model"
        args.embedding_similarity_gate = None
        args.blur_type = "gaussian"
        args.blur_strength = 15
        args.confidence_threshold = 0.5
        args.low_score_threshold = 0.2
        args.blur_classes = "license_plate,vehicle"
        args.tracker_params = '{"distance_gate":0.55}'
        args.tracker = None
        args.use_sahi = True
        args.sahi_overlap = 0.3
        args.inference_size = 2048
        args.offline_linker = None
        args.batch_size = 12
        args.video_codec = "hevc"
        args.video_quality = 18
        args.config_debug = True
        args.log_level = "debug"

        result = get_config_for_args(args)
        assert result.model.name == "test_model"
        assert result.blur.type.value == "gaussian"
        assert result.blur.strength == 15
        assert result.detection.confidence_threshold == 0.5
        assert result.detection.low_score_threshold == 0.2
        assert result.detection.classes_to_blur == ["license_plate", "vehicle"]
        mock_set_config.assert_called_once()
        assert result.detection.use_sahi is True
        assert result.detection.sahi_overlap_ratio == 0.3
        assert result.detection.inference_size == 2048
        assert result.detection.batch_size == 12
        assert result.tracking.params["distance_gate"] == 0.55
        assert result.video.codec == "hevc"
        assert result.video.quality == 18
        assert result.debug is True
        assert result.log_level == "DEBUG"

    def test_get_config_missing_attributes(self):
        """Test get_config_for_args with missing attributes on args."""
        args = Mock()
        args.config = None
        args.model = None
        args.embedding_similarity_gate = None
        args.blur_type = None
        args.blur_strength = None
        args.confidence_threshold = None
        args.low_score_threshold = None
        args.tracker_params = None
        args.tracker = None
        args.use_sahi = None
        args.sahi_overlap = None
        args.inference_size = None
        args.offline_linker = None
        args.batch_size = None
        args.video_codec = None
        args.video_quality = None
        args.config_debug = None
        args.log_level = "INFO"

        result = get_config_for_args(args)
        assert isinstance(result, AnonymizerConfig)


class TestCreateConfig:
    """Test create_config function."""

    @patch("blur_cli.cli.create_config_template")
    def test_create_config_success(self, mock_create_template):
        """Test successful config creation."""
        args = Mock()
        args.output = "/path/to/config.toml"

        result = create_config(args)
        assert result == 0
        mock_create_template.assert_called_once_with(Path("/path/to/config.toml"))

    @patch("blur_cli.cli.create_config_template")
    def test_create_config_default_filename(self, mock_create_template):
        """Test config creation with default filename."""
        args = Mock()
        args.output = None

        result = create_config(args)
        assert result == 0
        mock_create_template.assert_called_once_with(Path("blur_config.toml"))

    @patch("blur_cli.cli.create_config_template", side_effect=Exception("Test error"))
    def test_create_config_exception_handling(self, mock_create_template):
        """Test create_config exception handling."""
        args = Mock()
        args.output = "/path/to/config.toml"

        result = create_config(args)
        assert result == 1


class TestListModels:
    """Test list_models function."""

    @patch("blur_cli.cli._get_models_dir")
    def test_list_models_no_directory(self, mock_get_dir):
        """Test list_models when models directory doesn't exist."""
        mock_models_path = Mock()
        mock_models_path.exists.return_value = False
        mock_get_dir.return_value = mock_models_path

        result = list_models()
        assert result == 1

    @patch("blur_cli.cli._get_models_dir")
    def test_list_models_no_onnx_files(self, mock_get_dir):
        """Test list_models when no ONNX files found."""
        mock_models_path = Mock()
        mock_models_path.exists.return_value = True
        mock_models_path.glob.return_value = []  # No ONNX files
        mock_get_dir.return_value = mock_models_path

        result = list_models()
        assert result == 1

    @patch("blur_cli.cli._get_models_dir")
    def test_list_models_success(self, mock_get_dir):
        """Test successful model listing."""

        # Create mock model files with __lt__ method for sorting
        class MockModelFile:
            def __init__(self, stem):
                self.stem = stem

            def __lt__(self, other):
                return self.stem < other.stem

            def stat(self):
                mock_stat = Mock()
                mock_stat.st_size = 123456
                return mock_stat

        mock_model1 = MockModelFile("model1")
        mock_model2 = MockModelFile("model2")

        mock_models_path = Mock()
        mock_models_path.exists.return_value = True
        mock_models_path.glob.return_value = [
            mock_model2,
            mock_model1,
        ]  # Reverse order to test sorting
        mock_get_dir.return_value = mock_models_path

        result = list_models()
        assert result == 0
        mock_models_path.glob.assert_called_once_with("*.onnx")
