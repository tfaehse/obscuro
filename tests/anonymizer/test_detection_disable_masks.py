from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from anonymizer.detection import Detector


def test_disable_masks_config():
    """Test that disable_masks configuration is respected."""
    with (
        patch("anonymizer.detection.model.ort.InferenceSession"),
        patch("pathlib.Path.exists", return_value=True),
    ):
        model_path = Path("fake_model.onnx")

        # Default behavior: masks enabled (disable_masks=False)
        detector = Detector(model_path)
        assert detector.disable_masks is False

        # Disabled explicitly
        detector_disabled = Detector(model_path, disable_masks=True)
        assert detector_disabled.disable_masks is True


def test_disable_masks_postprocess():
    """Test that masks are skipped in postprocess when disabled."""
    with (
        patch("anonymizer.detection.model.ort.InferenceSession"),
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "anonymizer.detection.core.load_model_metadata",
            return_value={"classes": ["person", "head"], "default_blur": ["person", "head"]},
        ),
    ):
        detector = Detector(
            Path("fake_model.onnx"),
            disable_masks=True,
            categories_to_blur=["person"],  # Ensure class 0 is wanted
        )

        # Mock mask manager
        detector.mask_manager = Mock()

        # Create dummy output with masks
        # [x, y, w, h, class0_logit, class1_logit, mask_coeff1, mask_coeff2]
        # Use high logit (10.0) for class 0 to ensure high confidence
        boxes = [
            [100.0, 100.0, 50.0, 50.0, 10.0, -10.0, 0.5, 0.5],
        ]

        outputs = [
            np.array(
                [boxes],
                dtype=np.float32,
            ).reshape(1, 8, 1)
        ]

        # Mock mask proto (batch, mask_dim, h, w)
        # mask_dim=2 to match coeffs
        mask_proto = np.zeros((1, 2, 160, 160), dtype=np.float32)
        # Append mask proto to outputs if that's how the model returns it (usually separate or part of it)
        # The detector code seems to separate them or expect them in a certain way.
        # Let's see _postprocess implementation details in `src/anonymizer/detection/core.py`.
        # It expects `outputs` list. Usually `outputs[0]` is detection+mask_coeffs, `outputs[1]` is mask_proto.

        outputs_with_proto = [outputs[0], mask_proto]

        metas = [
            {
                "scale": (1.0, 1.0),
                "pad": (0.0, 0.0),
                "original_shape": (640, 640),
            }
        ]

        # Run postprocess
        df = detector._postprocess(outputs_with_proto, metas)

        # Assertions
        assert df.height == 1
        # Check that mask manager was NOT called to store proto
        detector.mask_manager.store_mask_proto.assert_not_called()

        # Check that dataframe has empty/None masks
        # 'mask' column should contain None
        row = df.row(0, named=True)
        assert row["mask"] is None
