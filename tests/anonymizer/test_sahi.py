from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import polars as pl
import pytest

from anonymizer.detection.sahi import SahiDetector


class TestSahiDetector:
    @pytest.fixture
    def mock_session(self):
        with patch("anonymizer.detection.model.ort.InferenceSession") as mock:
            session = MagicMock()
            session.get_inputs.return_value = [MagicMock(name="images")]
            session.get_providers.return_value = ["CPUExecutionProvider"]
            mock.return_value = session
            yield session

    @pytest.fixture
    def detector(self, mock_session):
        with patch("pathlib.Path.exists", return_value=True):
            return SahiDetector("fake_model.onnx", sahi_overlap_ratio=0.2)

    def test_initialization(self, detector):
        assert detector.sahi_overlap_ratio == 0.2
        assert detector._sahi_model is None

    def test_ensure_sahi_model(self, detector):
        model = detector._ensure_sahi_model()
        assert model is not None
        assert detector._sahi_model is model
        # Second call returns same instance
        assert detector._ensure_sahi_model() is model

    def test_prepare_image_for_sahi_rgb(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        prepared = SahiDetector._prepare_image_for_sahi(img)
        assert prepared.shape == (100, 100, 3)
        assert np.array_equal(prepared, img)

    def test_prepare_image_for_sahi_grayscale(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        prepared = SahiDetector._prepare_image_for_sahi(img)
        assert prepared.shape == (100, 100, 3)

    def test_prepare_image_for_sahi_channel_first(self):
        img = np.zeros((3, 100, 100), dtype=np.uint8)
        prepared = SahiDetector._prepare_image_for_sahi(img)
        assert prepared.shape == (100, 100, 3)

    def test_downscale_for_sahi_no_scale(self, detector):
        detector.inference_size = 640
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        resized, scale = detector._downscale_for_sahi(img)
        assert scale == 1.0
        assert resized.shape == (480, 640, 3)

    def test_downscale_for_sahi_scaled(self, detector):
        detector.inference_size = 640
        img = np.zeros((1280, 1280, 3), dtype=np.uint8)
        resized, scale = detector._downscale_for_sahi(img)
        assert scale == 0.5
        assert resized.shape == (640, 640, 3)

    @patch("anonymizer.detection.sahi.slice_image")
    def test_predict_single_image(self, mock_slice, detector):
        img = np.zeros((640, 640, 3), dtype=np.uint8)

        # Mock slice_image to return one tile
        mock_slice.return_value = [
            {
                "image": np.zeros((320, 320, 3), dtype=np.uint8),
                "starting_pixel": [0, 0],
            }
        ]

        # Mock session run
        detector.session.run.return_value = [
            np.zeros((1, 84, 100), dtype=np.float32)  # YOLOv8 output shape
        ]

        # Mock postprocess to return a DataFrame
        with patch.object(detector, "_postprocess") as mock_post:
            mock_post.return_value = pl.DataFrame({"x1": [10]})

            df = detector._predict_single_image(img)

            assert not df.is_empty()
            mock_slice.assert_called_once()
            detector.session.run.assert_called()
            mock_post.assert_called_once()

    @patch("anonymizer.detection.sahi.slice_image")
    def test_predict_single_image_no_tiles(self, mock_slice, detector):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_slice.return_value = []

        df = detector._predict_single_image(img)
        assert df.is_empty()

    def test_object_predictions_to_dataframe(self, detector):
        # Mock ObjectPrediction from sahi
        mock_pred = Mock()
        mock_pred.bbox.to_xyxy.return_value = [10, 10, 50, 50]
        # Ensure shift attributes are 0, not Mocks
        mock_pred.bbox.shift_x = 0
        mock_pred.bbox.shift_y = 0
        mock_pred.score.value = 0.9
        mock_pred.category.id = 0

        predictions = [mock_pred]

        df = detector._object_predictions_to_dataframe(
            predictions, frame_width=640, frame_height=480
        )

        assert not df.is_empty()
        assert len(df) == 1
        assert df["x1"][0] == 10.0
        assert df["confidence"][0] == 0.9
        assert df["object_class"][0] == 0

    @patch("anonymizer.detection.sahi.get_video_info")
    @patch("anonymizer.detection.sahi.iter_frame_batches")
    def test_detect_from_path(self, mock_iter, mock_info, detector):
        mock_info.return_value = {"frame_count": 10}

        # Mock batches: list of (index, frame) tuples
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_iter.return_value = [[(0, frame), (1, frame)]]

        # Mock predict_single_image
        with patch.object(detector, "_predict_single_image") as mock_predict:
            mock_predict.return_value = pl.DataFrame(
                {"x1": [10], "frame": [0]}
            )  # frame col is ignored/overwritten?
            # Actually _predict_single_image returns DF without frame col usually, _iter_frame_sequence adds it

            # Wait, _iter_frame_sequence calls _predict_single_image
            # And _detect_from_path calls _iter_frame_sequence via list comp?
            # No, _detect_from_path calls _iter_frame_sequence explicitly?
            # Let's check source.
            # _detect_from_path calls _iter_frame_sequence inside the loop.

            df = detector._detect_from_path(Path("video.mp4"))

            assert not df.is_empty()
            assert "frame" in df.columns
            assert len(df) == 2  # 2 frames in batch, both return detections (mocked same return)
