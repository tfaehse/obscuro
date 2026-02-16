from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

from anonymizer.sahi_integration import SahiOnnxDetectionModel


class TestSahiOnnxDetectionModel:
    @pytest.fixture
    def mock_detector(self):
        detector = Mock()
        detector.session = Mock()
        detector.imgsz = 640
        input_mock = Mock()
        input_mock.name = "images"  # Set as attribute, not Mock
        detector.session.get_inputs.return_value = [input_mock]
        detector._postprocess = Mock(
            return_value=pl.DataFrame(
                {
                    "frame": [0],
                    "x1": [10.0],
                    "y1": [10.0],
                    "x2": [50.0],
                    "y2": [50.0],
                    "confidence": [0.9],
                    "object_class": [0],
                    "frame_width": [640],
                    "frame_height": [640],
                    "is_confident": [True],
                }
            )
        )
        return detector

    @pytest.fixture
    def sahi_model(self, mock_detector):
        return SahiOnnxDetectionModel(detector=mock_detector)

    def test_initialization(self, sahi_model, mock_detector):
        assert sahi_model.detector == mock_detector
        assert sahi_model.model == mock_detector.session
        assert sahi_model._input_name == "images"
        assert sahi_model.tile_batch_size == 1

    def test_set_model(self, sahi_model, mock_detector):
        new_session = Mock()
        input_mock = Mock()
        input_mock.name = "input"
        new_session.get_inputs.return_value = [input_mock]

        sahi_model.set_model(new_session)

        assert sahi_model.model == new_session
        assert sahi_model._input_name == "input"

    def test_set_model_no_inputs(self, sahi_model):
        bad_session = Mock()
        bad_session.get_inputs.return_value = []

        with pytest.raises(ValueError, match="ONNX session has no input tensors"):
            sahi_model.set_model(bad_session)

    def test_perform_inference(self, sahi_model, mock_detector):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_detector.session.run.return_value = [np.zeros((1, 84, 100))]

        sahi_model.perform_inference(image)

        assert sahi_model._last_dataframe is not None
        assert not sahi_model._last_dataframe.is_empty()
        mock_detector.session.run.assert_called_once()

    def test_perform_inference_not_initialized(self, sahi_model):
        sahi_model.model = None
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="SAHI model is not initialised"):
            sahi_model.perform_inference(image)

    def test_predict_full_frame(self, sahi_model, mock_detector):
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        mock_detector.session.run.return_value = [np.zeros((1, 84, 100))]

        predictions = sahi_model._predict_full_frame(image)

        assert isinstance(predictions, list)

    def test_ensure_dataframe(self, sahi_model):
        # Test with DataFrame
        df = pl.DataFrame({"x1": [1.0]})
        sahi_model._original_predictions = df
        result = sahi_model._ensure_dataframe()
        assert result.equals(df)

        # Test with None
        sahi_model._original_predictions = None
        result = sahi_model._ensure_dataframe()
        assert result.is_empty()

        # Non-DataFrame payloads are no longer supported.
        sahi_model._original_predictions = [{"x1": 1.0}]
        with pytest.raises(TypeError):
            sahi_model._ensure_dataframe()

    def test_empty_dataframe(self):
        df = SahiOnnxDetectionModel._empty_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert df.is_empty()
        assert "frame" in df.columns
        assert "x1" in df.columns
        assert "confidence" in df.columns

    def test_create_object_prediction_list_basic(self, sahi_model, mock_detector):
        sahi_model._original_predictions = pl.DataFrame(
            {
                "frame": [0],
                "x1": [10.0],
                "y1": [10.0],
                "x2": [50.0],
                "y2": [50.0],
                "confidence": [0.9],
                "object_class": [0],
                "frame_width": [640],
                "frame_height": [480],
            }
        )

        sahi_model._create_object_prediction_list_from_original_predictions(
            shift_amount_list=[[0, 0]], full_shape_list=[[480, 640]]
        )

        assert sahi_model._object_prediction_list_per_image is not None
        assert len(sahi_model._object_prediction_list_per_image) == 1
        assert len(sahi_model._object_prediction_list_per_image[0]) == 1

    def test_create_object_prediction_list_with_shift(self, sahi_model):
        sahi_model._original_predictions = pl.DataFrame(
            {
                "frame": [0],
                "x1": [10.0],
                "y1": [10.0],
                "x2": [50.0],
                "y2": [50.0],
                "confidence": [0.9],
                "object_class": [1],
                "frame_width": [640],
                "frame_height": [480],
            }
        )

        sahi_model._create_object_prediction_list_from_original_predictions(
            shift_amount_list=[[100, 100]], full_shape_list=[[480, 640]]
        )

        predictions = sahi_model._object_prediction_list_per_image[0]
        assert len(predictions) == 1
        # Shift is applied to bbox, check it exists
        assert hasattr(predictions[0], "bbox")

    def test_create_object_prediction_list_empty(self, sahi_model):
        sahi_model._original_predictions = pl.DataFrame(
            {
                "frame": [],
                "x1": [],
                "y1": [],
                "x2": [],
                "y2": [],
                "confidence": [],
                "object_class": [],
                "frame_width": [],
                "frame_height": [],
            }
        )

        sahi_model._create_object_prediction_list_from_original_predictions()

        assert sahi_model._object_prediction_list_per_image == [[]]

    def test_last_dataframe_property(self, sahi_model):
        df = pl.DataFrame({"x1": [1.0]})
        sahi_model._last_dataframe = df
        assert sahi_model.last_dataframe.equals(df)

    def test_category_mapping(self, mock_detector):
        custom_mapping = {"0": "person", "1": "car"}
        model = SahiOnnxDetectionModel(detector=mock_detector, category_mapping=custom_mapping)

        assert "0" in model.category_mapping
        assert model.category_mapping["0"] == "person"

    @patch("anonymizer.sahi_integration.slice_image")
    def test_run_sliced_predictions_no_slices(self, mock_slice, sahi_model):
        mock_slice.return_value = Mock(sliced_image_list=[])
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        predictions = sahi_model._run_sliced_predictions(
            image,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        assert predictions == []
        assert sahi_model._object_prediction_list_per_image == [[]]

    @patch("anonymizer.sahi_integration.slice_image")
    def test_run_sliced_predictions_with_tiles(self, mock_slice, sahi_model, mock_detector):
        tile0 = Mock()
        tile0.image = np.zeros((320, 320, 3), dtype=np.uint8)
        tile0.starting_pixel = [0, 0]
        tile1 = Mock()
        tile1.image = np.zeros((320, 320, 3), dtype=np.uint8)
        tile1.starting_pixel = [320, 0]
        mock_slice.return_value = Mock(sliced_image_list=[tile0, tile1])
        mock_detector.session.run.return_value = [np.zeros((2, 84, 100))]

        image = np.zeros((640, 640, 3), dtype=np.uint8)

        predictions = sahi_model._run_sliced_predictions(
            image,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        assert isinstance(predictions, list)
        mock_detector.session.run.assert_called()
