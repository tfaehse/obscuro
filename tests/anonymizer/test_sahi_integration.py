from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from sahi.prediction import ObjectPrediction

from anonymizer.detection import Detector, SahiDetector


def _make_prediction(x1, y1, x2, y2, score=0.9, category_id=0, shift=(0, 0)):
    return ObjectPrediction(
        bbox=[x1, y1, x2, y2],
        category_id=category_id,
        category_name="cls",
        score=score,
        shift_amount=list(shift),
        full_shape=[100, 100],
    )


def test_downscale_for_sahi_respects_inference_size():
    detector = SahiDetector.__new__(SahiDetector)
    detector.inference_size = 256
    image = np.zeros((400, 200, 3), dtype=np.uint8)
    resized, scale = SahiDetector._downscale_for_sahi(detector, image)
    assert resized.shape[0] == 256
    assert resized.shape[1] == 128
    assert scale == pytest.approx(256 / 400)


def test_downscale_for_sahi_skips_small_frames():
    detector = SahiDetector.__new__(SahiDetector)
    detector.inference_size = 512
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    resized, scale = SahiDetector._downscale_for_sahi(detector, image)
    assert resized is image
    assert scale == 1.0


def test_object_predictions_dataframe_applies_shifts_and_scaling():
    detector, _fake_session = _build_detector_with_session()
    predictions = [
        _make_prediction(10, 20, 30, 40, score=0.8, category_id=0, shift=(5, 7)),
    ]
    df = detector._object_predictions_to_dataframe(
        predictions,
        frame_width=100,
        frame_height=100,
        original_width=200,
        original_height=200,
        scale_factor=0.5,
    )
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["x1"] == pytest.approx((10 + 5) * 2)
    assert row["y1"] == pytest.approx((20 + 7) * 2)
    assert row["confidence"] == pytest.approx(0.8)
    assert row["frame_width"] == 200
    assert row["frame_height"] == 200


def _build_detector_with_session(session_output=None):
    if session_output is None:
        session_output = [
            np.array(
                [
                    [
                        [50.0, 50.0, 10.0, 10.0, 0.9, 0.1],
                    ]
                ],
                dtype=np.float32,
            )
        ]

    with (
        patch(
            "onnxruntime.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ),
        patch("onnxruntime.InferenceSession") as mock_session,
        patch(
            "anonymizer.detection.core.load_model_metadata",
            return_value={"classes": ["cls"], "default_blur": ["cls"]},
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
        detector = Detector(Path("fake_model.onnx"))

    return detector, fake_session
