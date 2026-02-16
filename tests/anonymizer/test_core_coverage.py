from unittest.mock import Mock, patch

import pytest

from anonymizer.core import Anonymizer


class TestAnonymizerCoverage:
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.model.path = "/fake/model.onnx"
        config.detection.confidence_threshold = 0.5
        config.detection.low_score_threshold = 0.3
        config.detection.batch_size = 1
        config.detection.inference_size = 640
        config.detection.sahi_overlap_ratio = 0.2
        config.detection.classes_to_blur = []
        config.tracking.type = "dummy"
        # Make blur.type look like an enum with value
        blur_type_mock = Mock()
        blur_type_mock.value = "pixelate"
        config.blur.type = blur_type_mock
        config.blur.strength = 10
        config.get_tracker_kwargs.return_value = {}
        config.model.get_execution_providers.return_value = []
        return config

    @patch("anonymizer.core.Detector")
    @patch("anonymizer.core.TrackerFactory")
    @patch("anonymizer.core.Blurrer")
    @patch("anonymizer.core.get_config")
    def test_anonymizer_init_with_config(
        self, mock_get_config, mock_blurrer, mock_tracker_factory, mock_detector, mock_config
    ):
        mock_tracker_factory.get.return_value = Mock()

        anonymizer = Anonymizer(config=mock_config)

        assert anonymizer.config == mock_config
        mock_detector.assert_called_once()
        mock_tracker_factory.get.assert_called_once()

    @patch("anonymizer.core.Detector")
    @patch("anonymizer.core.TrackerFactory")
    @patch("anonymizer.core.Blurrer")
    @patch("anonymizer.core.get_config")
    def test_anonymizer_init_without_config(
        self, mock_get_config, mock_blurrer, mock_tracker_factory, mock_detector, mock_config
    ):
        mock_get_config.return_value = mock_config
        mock_tracker_factory.get.return_value = Mock()

        anonymizer = Anonymizer()

        assert anonymizer.config == mock_config
        mock_get_config.assert_called_once()
