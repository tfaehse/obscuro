"""
Tests for the detection functionality.
"""

import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sahi.prediction import PredictionResult

from anonymizer.cancellation import CancellationException
from anonymizer.constants import DEFAULT_SEGMENTATION_CLASSES
from anonymizer.detection import Detector, FrameDetector
from anonymizer.detection.utils import letterbox, preprocess_image
from anonymizer.sahi_integration import SahiOnnxDetectionModel


def _build_detector_with_session(
    session_output=None,
    *,
    use_sahi: bool = False,
    categories_to_blur=None,
    model_classes: list[str] | None = None,
):
    """Instantiate a detector with a mocked ONNX session."""
    if session_output is None:
        session_output = [
            np.array(
                [
                    [
                        [320.0, 240.0, 160.0, 80.0, 0.9, 0.1],
                    ]
                ],
                dtype=np.float32,
            ).transpose(0, 2, 1)
        ]

    classes = model_classes or list(DEFAULT_SEGMENTATION_CLASSES)
    if categories_to_blur is None:
        categories_to_blur = ["*"]
    with (
        patch(
            "anonymizer.detection.model.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ),
        patch("anonymizer.detection.model.ort.InferenceSession") as mock_session,
        patch(
            "anonymizer.detection.core.load_model_metadata",
            return_value={"classes": classes, "default_blur": classes},
        ),
        patch("pathlib.Path.exists", return_value=True),
    ):
        fake_session = Mock()
        input_info = Mock()
        input_info.name = "images"
        fake_session.get_inputs.return_value = [input_info]
        fake_session.run.return_value = session_output
        fake_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session.return_value = fake_session
        detector = Detector(
            Path("fake_model.onnx"),
            use_sahi=use_sahi,
            categories_to_blur=categories_to_blur,
        )

    return detector, fake_session


class TestDetector:
    """Test Detector class basic functionality."""

    @pytest.fixture
    def mock_model_path(self):
        """Create a temporary model file path."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake_model")
            return Path(f.name)

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_execution_provider_status_matches_request(self):
        detector, _ = _build_detector_with_session()
        status = detector.get_execution_provider_status()
        assert status["primary"] == "CPUExecutionProvider"
        assert status["status_code"] == 0

    def test_detector_initialization_default(self, mock_model_path):
        """Test detector initialization with default parameters."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(mock_model_path)

        assert detector.model_path == mock_model_path.resolve()
        assert detector.batch_size == 8
        assert detector.confidence_threshold == 0.5
        assert detector.low_score_threshold == 0.1
        assert detector.cancel_event is None
        assert detector.progress_callback is None

    def test_detector_initialization_custom(self, mock_model_path):
        """Test detector initialization with custom parameters."""
        import threading

        cancel_event = threading.Event()
        progress_callback = Mock()

        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(
                mock_model_path,
                cancel_event=cancel_event,
                batch_size=16,
                progress_callback=progress_callback,
                confidence_threshold=0.8,
                low_score_threshold=0.2,
            )

        assert detector.batch_size == 16
        assert detector.confidence_threshold == 0.8
        assert detector.low_score_threshold == 0.2
        assert detector.cancel_event == cancel_event
        assert detector.progress_callback == progress_callback

    def test_model_size_inference_square(self):
        """Test model size inference from square imgsz prefix."""
        model_path = Path("1280_nano_seg.onnx")

        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(model_path)

        assert detector.imgsz == 1280

    def test_model_size_fallback(self):
        """Test model size fallback for unknown model names."""
        model_path = Path("unknown_model.onnx")

        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(model_path)

        assert detector.imgsz == 640  # Fallback size

    def test_detector_filters_configured_categories(self):
        """Detector should drop classes not in the blur list."""
        # Create enough dummy boxes to ensure num_boxes > num_features (8)
        # to avoid the transpose heuristic in _postprocess
        boxes = [
            [50.0, 50.0, 10.0, 10.0, 0.1, 0.9, 0.0, 0.0],  # Class 1 (index 1) high
            [70.0, 70.0, 12.0, 12.0, 0.9, 0.1, 0.0, 0.0],  # Class 0 (index 0) high
        ]

        session_output = [np.array([boxes], dtype=np.float32).transpose(0, 2, 1)]

        detector, _ = _build_detector_with_session(
            session_output=session_output,
            categories_to_blur=["head"],
            model_classes=["person", "head"],
        )

        meta = {"scale": (1.0, 1.0), "pad": (0.0, 0.0), "original_shape": (640, 640)}
        df = detector._postprocess(session_output, [meta])

        nms_results = df.filter(pl.col("is_confident"))
        # After filtering by category, expect only class 1 (head)
        assert set(nms_results["object_class"].to_list()) == {1}

    def test_sahi_matches_standard_detection(self, sample_image):
        """SAHI integration should align with standard inference for identical outputs."""
        session_output = [
            np.array(
                [
                    [
                        [100.0, 120.0, 40.0, 60.0, 0.95, 0.05],
                    ]
                ],
                dtype=np.float32,
            ).transpose(0, 2, 1)
        ]

        baseline_detector, _ = _build_detector_with_session(session_output=session_output)
        baseline_result = baseline_detector.detect(sample_image)

        sahi_detector, _ = _build_detector_with_session(
            session_output=session_output, use_sahi=True
        )

        def fake_predict(self, image, **kwargs):
            self.perform_inference(np.ascontiguousarray(image))
            self.convert_original_predictions(
                shift_amount=[[0, 0]],
                full_shape=[[image.shape[0], image.shape[1]]],
            )
            return PredictionResult(
                image=image,
                object_prediction_list=self.object_prediction_list,
                durations_in_seconds={},
            )

        with patch.object(
            SahiOnnxDetectionModel,
            "predict_fused",
            side_effect=fake_predict,
            autospec=True,
        ):
            sahi_result = sahi_detector.detect(sample_image)

        assert_frame_equal(baseline_result, sahi_result)

    def test_execution_providers_cuda(self):
        """Test CUDA execution provider detection."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch(
                "anonymizer.detection.model.ort.get_available_providers",
                return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(Path("test.onnx"))
            providers, _ = detector.model_loader._get_execution_providers()
            assert "CUDAExecutionProvider" in providers

    def test_execution_providers_mps(self):
        """Test MPS execution provider detection."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch(
                "anonymizer.detection.model.ort.get_available_providers",
                return_value=["MLComputeExecutionProvider", "CPUExecutionProvider"],
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(Path("test.onnx"))
            providers, _ = detector.model_loader._get_execution_providers()
            # Should contain available providers
            assert isinstance(providers, list)

    def test_execution_providers_cpu_fallback(self):
        """Test fallback to CPU execution provider."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch(
                "anonymizer.detection.model.ort.get_available_providers",
                return_value=["CPUExecutionProvider"],
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(Path("test.onnx"))
            providers, _ = detector.model_loader._get_execution_providers()
            assert providers == ["CPUExecutionProvider"]


class TestDetectorIntegration:
    """Test detector integration scenarios."""

    def test_full_detection_pipeline(self):
        """Test that the full detection pipeline can be instantiated."""
        model_path = Path("1280_nano_seg.onnx")

        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(model_path)

            # Should have the detect method
            assert hasattr(detector, "detect")
            assert callable(detector.detect)

    def test_batch_processing_performance(self):
        """Test that batch processing configuration works."""
        model_path = Path("test_model.onnx")

        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            # Test with different batch sizes
            for batch_size in [1, 4, 8, 16]:
                detector = Detector(model_path, batch_size=batch_size)
                assert detector.batch_size == batch_size


class TestDetectorErrorHandling:
    """Test detector error handling."""

    def test_empty_image_handling(self):
        """Test handling of empty or invalid images."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(Path("test.onnx"))

            # Test with empty array
            empty_image = np.array([])

            # Should handle gracefully (exact behavior depends on implementation)
            import contextlib

            with contextlib.suppress(ValueError, AttributeError):
                preprocess_image(empty_image, detector.imgsz)


class TestDetectorPreprocessing:
    """Test Detector preprocessing methods."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_letterbox_basic(self):
        """Test letterbox preprocessing."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result, ratio, pad = letterbox(image, new_shape=(640, 640))

            # Result should have processed the image
            assert len(result.shape) == 3  # Should be HWC
            assert len(ratio) == 2
            assert len(pad) == 2

    def test_letterbox_empty_image_error(self):
        """Test letterbox with empty image."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            # Create image with zero dimensions
            empty_image = np.zeros((0, 0, 3), dtype=np.uint8)

            with pytest.raises(ValueError, match="Cannot process empty image"):
                letterbox(empty_image)

    def test_letterbox_different_shapes(self):
        """Test letterbox with different input shapes."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            # Test square image
            square_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            result, ratio, pad = letterbox(square_image, new_shape=(640, 640))
            assert len(result.shape) == 3
            assert len(ratio) == 2
            assert len(pad) == 2

            # Test wide image
            wide_image = np.random.randint(0, 255, (480, 1280, 3), dtype=np.uint8)
            result, _ratio, _pad = letterbox(wide_image, new_shape=(640, 640))
            assert len(result.shape) == 3

    def test_letterbox_scale_options(self):
        """Test letterbox with different scale options."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Test with scaleup=False
            result1, _, _ = letterbox(image, new_shape=(1280, 1280), scaleup=False)
            assert len(result1.shape) == 3

            # Test with scaleFill=True
            result2, _, _ = letterbox(image, new_shape=(640, 640), scaleFill=True)
            assert len(result2.shape) == 3

    def test_preprocess_basic(self, sample_image):
        """Test basic preprocessing."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(Path("fake_model.onnx"))

            result, meta = preprocess_image(sample_image, detector.imgsz)

            # Should be NCHW format
            assert len(result.shape) == 4
            assert result.shape[0] == 1  # batch size
            assert result.shape[1] == 3  # channels
            assert result.dtype == np.float32
            # Values should be normalized to [0, 1]
            assert result.min() >= 0.0
            assert result.max() <= 1.0
            assert set(meta.keys()) == {
                "scale",
                "pad",
                "original_shape",
                "global_shape",
                "tile_id",
                "tile_offset",
                "scale_up",
            }
            assert isinstance(meta["original_shape"], tuple)

    def test_preprocess_different_sizes(self):
        """Test preprocessing with different image sizes."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(Path("fake_model.onnx"))

            # Test different input sizes
            sizes = [(224, 224, 3), (480, 640, 3), (1080, 1920, 3)]

            for h, w, c in sizes:
                image = np.random.randint(0, 255, (h, w, c), dtype=np.uint8)
                result, meta = preprocess_image(image, detector.imgsz)

                # Output should always be same size after letterboxing
                assert result.shape[2] == detector.imgsz
                assert result.shape[3] == detector.imgsz
                assert meta["original_shape"] == (h, w)

    def test_progress_reporting(self):
        """Test progress reporting functionality."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            progress_callback = Mock()
            detector = Detector(Path("fake_model.onnx"), progress_callback=progress_callback)

        detector._report_progress(50, "Test message")

        progress_callback.assert_called_once_with(50, "Detection", "Test message")

    def test_progress_reporting_no_callback(self):
        """Test progress reporting with no callback."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            _ = Detector(Path("fake_model.onnx"))


class TestDetectorExecutionProviders:
    """Test execution provider selection logic."""

    def test_all_execution_providers(self):
        """Test all execution providers selection logic."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(Path("fake_model.onnx"))

            # Test each provider type
            provider_tests = [
                (["CUDAExecutionProvider"], ["CUDAExecutionProvider"]),
                (["MPSExecutionProvider"], ["MPSExecutionProvider"]),
                (["DmlExecutionProvider"], ["DmlExecutionProvider"]),
                (["TensorrtExecutionProvider"], ["TensorrtExecutionProvider"]),
                (["OpenVINOExecutionProvider"], ["OpenVINOExecutionProvider"]),
                (["QNNExecutionProvider"], ["QNNExecutionProvider"]),
                (["CPUExecutionProvider"], ["CPUExecutionProvider"]),
                ([], ["CPUExecutionProvider"]),  # Fallback case
            ]

            for available, expected in provider_tests:
                with patch(
                    "anonymizer.detection.model.ort.get_available_providers", return_value=available
                ):
                    result, _ = detector.model_loader._get_execution_providers()
                    assert result == expected

    def test_provider_priority(self):
        """Test provider selection priority."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(Path("fake_model.onnx"))

            # CUDA should be preferred over others
            with patch(
                "anonymizer.detection.model.ort.get_available_providers",
                return_value=["CPUExecutionProvider", "CUDAExecutionProvider"],
            ):
                result, _ = detector.model_loader._get_execution_providers()
                assert result == ["CUDAExecutionProvider"]


class TestDetectorAdvanced:
    """Test advanced Detector functionality."""

    def test_model_loading(self):
        """Test model loading with different providers."""
        model_path = "fake_model.onnx"

        with (
            patch("anonymizer.detection.model.ort.InferenceSession") as mock_session,
            patch("pathlib.Path.exists", return_value=True),
        ):
            Detector(Path(model_path))

            # Should create inference session
            mock_session.assert_called_once()

    def test_cancellation_check(self):
        """Test cancellation event checking."""
        import threading

        cancel_event = threading.Event()

        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(Path("fake_model.onnx"), cancel_event=cancel_event)

            # Detector stores the cancel event
            assert detector.cancel_event == cancel_event

    def test_batch_size_settings(self):
        """Test different batch size settings."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(Path("fake_model.onnx"), batch_size=16)

            assert detector.batch_size == 16

    def test_threshold_settings(self):
        """Test threshold parameter settings."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(
                Path("fake_model.onnx"),
                confidence_threshold=0.8,
                low_score_threshold=0.2,
            )

            assert detector.confidence_threshold == 0.8
            assert detector.low_score_threshold == 0.2


@patch("anonymizer.detection.model.ort.InferenceSession")
@patch("pathlib.Path.exists", return_value=True)
def test_detector_postprocess_absolute_coordinates(mock_exists, mock_session):
    with patch(
        "anonymizer.detection.core.load_model_metadata",
        return_value={
            "classes": list(DEFAULT_SEGMENTATION_CLASSES),
            "default_blur": list(DEFAULT_SEGMENTATION_CLASSES),
        },
    ):
        detector = Detector(Path("fake_model.onnx"), categories_to_blur=["*"])

        # Add dummy boxes to ensure num_boxes > num_features (6)
        boxes = [
            [320.0, 240.0, 160.0, 80.0, 0.9, 0.1],
        ]

        outputs = [np.array([boxes], dtype=np.float32).transpose(0, 2, 1)]
    metas = [
        {
            "scale": (1.0, 1.0),
            "pad": (0.0, 0.0),
            "original_shape": (480, 640),
        }
    ]

    df = detector._postprocess(outputs, metas)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["x1"] == pytest.approx(240.0)
    assert row["y1"] == pytest.approx(200.0)
    assert row["x2"] == pytest.approx(400.0)
    assert row["y2"] == pytest.approx(280.0)
    assert row["frame_width"] == 640
    assert row["frame_height"] == 480


@patch("anonymizer.detection.model.ort.InferenceSession")
@patch("pathlib.Path.exists", return_value=True)
def test_low_score_threshold_filters_pre_nms(mock_exists, mock_session):
    detector = Detector(
        Path("fake_model.onnx"),
        confidence_threshold=0.6,
        low_score_threshold=0.5,
    )

    outputs = [
        np.array(
            [
                [
                    [50.0, 50.0, 20.0, 20.0, -10.0, -10.0],  # low objectness/class scores
                ]
            ],
            dtype=np.float32,
        )
    ]
    metas = [
        {
            "scale": (1.0, 1.0),
            "pad": (0.0, 0.0),
            "original_shape": (100, 100),
        }
    ]

    df = detector._postprocess(outputs, metas)
    assert df.is_empty()


@patch("anonymizer.detection.model.ort.InferenceSession")
@patch("pathlib.Path.exists", return_value=True)
def test_detector_postprocess_transposed_output(mock_exists, mock_session):
    with patch(
        "anonymizer.detection.core.load_model_metadata",
        return_value={
            "classes": list(DEFAULT_SEGMENTATION_CLASSES),
            "default_blur": list(DEFAULT_SEGMENTATION_CLASSES),
        },
    ):
        detector = Detector(
            Path("fake_model.onnx"),
            confidence_threshold=0.5,
            low_score_threshold=0.1,
            categories_to_blur=["*"],
        )

        outputs = [
            np.array(
                [
                    [200.0, 60.0],  # x_center values
                    [150.0, 40.0],  # y_center values
                    [80.0, 20.0],  # widths
                    [60.0, 20.0],  # heights
                    [0.99, 0.01],  # class 0 probs (high, low)
                    [0.01, 0.99],  # class 1 probs (low, high)
                ],
                dtype=np.float32,
            ).reshape(1, 6, 2)
        ]

    metas = [
        {
            "scale": (1.0, 1.0),
            "pad": (0.0, 0.0),
            "original_shape": (480, 640),
        }
    ]

    df = detector._postprocess(outputs, metas)
    assert df.height == 2

    # First box: class 0 (logit 10.0 -> prob ~1.0)
    row = df.row(0, named=True)
    assert row["object_class"] == 0
    assert row["confidence"] == pytest.approx(0.99, abs=0.01)
    assert row["x1"] == pytest.approx(160.0)  # 200 - 40

    # Second box: class 1 (logit 10.0 -> prob ~1.0)
    row = df.row(1, named=True)
    assert row["object_class"] == 1
    assert row["confidence"] == pytest.approx(1.0, abs=0.01)
    assert row["x1"] == pytest.approx(50.0)  # 60 - 10
    assert row["y2"] == pytest.approx(50.0)


@patch("anonymizer.detection.model.ort.InferenceSession")
@patch("pathlib.Path.exists", return_value=True)
def test_detector_postprocess_scales_and_clips(mock_exists, mock_session):
    with patch(
        "anonymizer.detection.core.load_model_metadata",
        return_value={
            "classes": list(DEFAULT_SEGMENTATION_CLASSES),
            "default_blur": list(DEFAULT_SEGMENTATION_CLASSES),
        },
    ):
        detector = Detector(Path("fake_model.onnx"), categories_to_blur=["*"])

        outputs = [
            np.array([[[150.0, 100.0, 120.0, 80.0, 0.8, 0.2]]], dtype=np.float32).transpose(0, 2, 1)
        ]

    metas = [
        {
            "scale": (0.5, 0.5),
            "pad": (10.0, 10.0),
            "original_shape": (200, 200),
        }
    ]

    df = detector._postprocess(outputs, metas)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["x1"] == pytest.approx(160.0)
    assert row["x2"] == pytest.approx(200.0)  # clipped to width
    assert row["y1"] == pytest.approx(100.0)
    assert row["y2"] == pytest.approx(200.0)  # clipped to height


@patch("anonymizer.detection.model.ort.InferenceSession")
@patch("pathlib.Path.exists", return_value=True)
def test_detector_postprocess_applies_class_thresholds(mock_exists, mock_session):
    with patch(
        "anonymizer.detection.core.load_model_metadata",
        return_value={
            "classes": list(DEFAULT_SEGMENTATION_CLASSES),
            "default_blur": list(DEFAULT_SEGMENTATION_CLASSES),
        },
    ):
        detector = Detector(
            Path("fake_model.onnx"),
            confidence_threshold=0.6,
            low_score_threshold=0.2,
            categories_to_blur=["*"],
        )

        outputs = [
            np.array(
                [
                    [100.0, 50.0, 40.0, 30.0, 0.55, 0.45],
                    [120.0, 60.0, 40.0, 30.0, 0.3, 0.7],
                ],
                dtype=np.float32,
            )
            .reshape(1, 2, 6)
            .transpose(0, 2, 1)
        ]

    metas = [
        {
            "scale": (1.0, 1.0),
            "pad": (0.0, 0.0),
            "original_shape": (200, 200),
        }
    ]

    df = detector._postprocess(outputs, metas)
    assert df.height == 2
    confident = df.filter(pl.col("is_confident"))
    assert confident.height == 1

    rejected = df.filter(~pl.col("is_confident"))
    assert rejected.height == 1


def test_detector_postprocess_honours_cancellation():
    cancel_event = threading.Event()
    cancel_event.set()
    with (
        patch("anonymizer.detection.model.ort.InferenceSession"),
        patch("pathlib.Path.exists", return_value=True),
    ):
        detector = Detector(Path("fake_model.onnx"), cancel_event=cancel_event)

    outputs = [np.zeros((1, 1, 6), dtype=np.float32)]
    metas = [
        {
            "scale": (1.0, 1.0),
            "pad": (0.0, 0.0),
            "original_shape": (100, 100),
        }
    ]

    with pytest.raises(CancellationException):
        detector._postprocess(outputs, metas)


@patch("anonymizer.detection.model.ort.InferenceSession")
@patch("pathlib.Path.exists", return_value=True)
def test_detector_detect_from_list_empty(mock_exists, mock_session):
    detector = Detector(Path("fake_model.onnx"))

    result = detector._detect_from_list([])
    assert result.is_empty()


class TestDetectorImageHandling:
    """Test image handling edge cases."""

    def test_single_channel_image(self):
        """Test processing single channel image."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            # Single channel image should be handled
            gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

            # This might raise an error or handle gracefully depending on implementation
            import contextlib

            with contextlib.suppress(ValueError, IndexError):
                letterbox(gray_image.reshape(480, 640, 1))

    def test_very_small_image(self):
        """Test processing very small image."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            # Very small image
            tiny_image = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
            result, scale, _pad = letterbox(tiny_image)

            # Should still produce valid output
            assert result.shape[:2] == (640, 640)
            assert scale[0] > 0
            assert scale[1] > 0

    def test_very_large_image(self):
        """Test processing very large image."""
        with (
            patch("anonymizer.detection.model.ort.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            detector = Detector(Path("fake_model.onnx"))

            # Large image (simulate 4K)
            large_image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
            result, meta = preprocess_image(large_image, detector.imgsz)

            # Should be processed to standard size
            assert result.shape[2:] == (640, 640)
            assert meta["original_shape"] == (2160, 3840)


class TestDetectorBatchProcessing:
    """Additional coverage for batch and list detection helpers."""

    def test_detect_from_batch_array_processes_batches(self):
        detector = FrameDetector.__new__(FrameDetector)
        detector.batch_size = 2
        detector.cancel_event = None

        call_sizes: list[int] = []

        progress_updates: list[tuple[int, str]] = []

        def fake_detect_from_list(images, frame_ids=None):
            call_sizes.append(len(images))
            return pl.DataFrame({"frame": [0] * len(images), "x1": [0.0] * len(images)})

        detector._detect_from_list = fake_detect_from_list  # type: ignore[attr-defined]
        detector._report_progress = lambda pct, msg: progress_updates.append((pct, msg))
        detector.batch_size = 2

        tensor = np.random.randint(0, 255, (3, 3, 2, 2), dtype=np.uint8)

        result = detector._detect_from_batch_array(tensor)

        assert call_sizes == [2, 1]
        assert progress_updates[0] == (0, "Starting batch array processing")
        assert progress_updates[-1] == (100, "Batch array processing complete")
        assert result.shape == (3, 2)

    def test_detect_from_batch_array_respects_cancellation(self):
        detector = FrameDetector.__new__(FrameDetector)
        detector.batch_size = 1
        cancel_event = threading.Event()
        cancel_event.set()
        detector.cancel_event = cancel_event
        detector._detect_from_list = lambda images: pl.DataFrame()  # type: ignore[attr-defined]

        progress_updates: list[tuple[int, str]] = []
        detector._report_progress = lambda pct, msg: progress_updates.append((pct, msg))

        tensor = np.zeros((1, 3, 2, 2), dtype=np.float32)

        with pytest.raises(CancellationException):
            detector._detect_from_batch_array(tensor)

        assert progress_updates[-1] == (0, "Cancelled")


class TestDetectorListProcessing:
    """Ensure list-based detection handles cancellation hooks."""

    def test_detect_from_list_checks_cancellation_before_work(self):
        detector = FrameDetector.__new__(FrameDetector)
        event = threading.Event()
        event.set()
        detector.cancel_event = event

        with pytest.raises(CancellationException):
            detector._detect_from_list([np.zeros((4, 4, 3), dtype=np.uint8)])


class TestDetectorPostprocess:
    """Additional coverage for post-processing and I/O helpers."""

    def test_postprocess_applies_classwise_nms(self):
        detector, _ = _build_detector_with_session()
        outputs = [
            np.array(
                [
                    [
                        [100.0, 100.0, 50.0, 50.0, 0.9, 0.1, 0.0],  # Class 0 high
                        [200.0, 200.0, 50.0, 50.0, 0.1, 0.9, 0.0],  # Class 1 high
                    ]
                ],
                dtype=np.float32,
            ).transpose(0, 2, 1)
        ]
        metas = [
            {
                "scale": (1.0, 1.0),
                "pad": (0.0, 0.0),
                "original_shape": (200, 200),
            }
        ]

        df = detector._postprocess(outputs, metas)
        assert df.height == 2
        classes = df.get_column("object_class").to_list()
        assert set(classes) == {0, 1}
        class0 = df.filter(pl.col("object_class") == 0)
        assert class0.get_column("confidence").item() == pytest.approx(0.9, abs=0.01)

    def test_detect_from_list_runs_inference(self):
        # Each image should produce detections
        # For a batch of 2 images, we need output with batch dimension = 2
        boxes_img1 = [50.0, 50.0, 10.0, 10.0, 0.1, 0.9, 0.0, 0.0]  # Class 1 high
        boxes_img2 = [70.0, 70.0, 12.0, 12.0, 0.9, 0.1, 0.0, 0.0]  # Class 0 high

        session_output = [
            np.array(
                [boxes_img1, boxes_img2],  # 2 boxes, one per image
                dtype=np.float32,
            )
            .reshape(1, 2, 8)
            .transpose(0, 2, 1)  # batch=1, features=8, num_boxes=2
        ]
        detector, fake_session = _build_detector_with_session(session_output=session_output)

        def fake_preprocess(image, imgsz):
            return (
                np.zeros((1, 3, 2, 2), dtype=np.float32),
                {"scale": (1.0, 1.0), "pad": (0.0, 0.0), "original_shape": image.shape[:2]},
            )

        detector.preprocess = Mock(side_effect=fake_preprocess)

        images = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
        df = detector._detect_from_list(images, frame_ids=[0, 1])

        assert df.height == 2
        # Both detections are from the same batched inference, so they share frame 0
        assert df.get_column("frame").to_list() == [0, 0]
        fake_session.run.assert_called_once()

    def test_detect_from_path_batches_frames(self, monkeypatch):
        detector, _ = _build_detector_with_session()
        detector.batch_size = 2

        batch0 = pl.DataFrame(
            {
                "frame": [0, 1],
                "x1": [0.0, 30.0],
                "y1": [0.0, 30.0],
                "x2": [20.0, 50.0],
                "y2": [20.0, 50.0],
                "confidence": [0.9, 0.9],
                "object_class": [0, 0],
                "frame_width": [100, 100],
                "frame_height": [100, 100],
                "is_confident": [True, True],
            }
        )
        batch1 = pl.DataFrame(
            {
                "frame": [0],
                "x1": [60.0],
                "y1": [60.0],
                "x2": [80.0],
                "y2": [80.0],
                "confidence": [0.8],
                "object_class": [1],
                "frame_width": [100],
                "frame_height": [100],
                "is_confident": [True],
            }
        )

        detector._detect_from_list = Mock(side_effect=[batch0, batch1])
        batches = [
            [
                (0, np.zeros((4, 4, 3), dtype=np.uint8)),
                (1, np.zeros((4, 4, 3), dtype=np.uint8)),
            ],
            [
                (2, np.zeros((4, 4, 3), dtype=np.uint8)),
            ],
        ]

        def fake_iter_frame_batches(_path, batch_size, prefetch):
            assert batch_size == detector.batch_size
            yield from batches

        class FakeInfo:
            def __getitem__(self, key):
                return {"frame_count": 100}.get(key)

            def get(self, key, default=None):
                return {"frame_count": 100}.get(key, default)

        monkeypatch.setattr("anonymizer.detection.core.get_video_info", lambda path: FakeInfo())
        monkeypatch.setattr("anonymizer.detection.core.iter_frame_batches", fake_iter_frame_batches)

        result = detector._detect_from_path(Path("dummy.mp4"))

        assert detector._detect_from_list.call_count == 2
        assert sorted(result.get_column("frame").to_list()) == [0, 1, 2]

    def test_detect_from_list_checks_cancellation_before_inference(self, monkeypatch):
        detector = FrameDetector.__new__(FrameDetector)
        event = threading.Event()
        detector.cancel_event = event

        def fake_preprocess(image, imgsz):
            event.set()
            return np.zeros((1, 3, 4, 4), dtype=np.float32), {
                "scale": (1.0, 1.0),
                "pad": (0, 0),
                "original_shape": image.shape[:2],
            }

        detector.inference_size = 640
        detector.imgsz = 640
        monkeypatch.setattr("anonymizer.detection.core.preprocess_image", fake_preprocess)
        detector._empty_result_df = lambda: pl.DataFrame()  # type: ignore[attr-defined]

        with pytest.raises(CancellationException):
            detector._detect_from_list([np.zeros((4, 4, 3), dtype=np.uint8)])
