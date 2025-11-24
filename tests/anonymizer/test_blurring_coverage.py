from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from anonymizer.blurring import Blurrer, convert_relative_to_absolute_rois


class TestBlurringCoverage:
    @pytest.fixture
    def blurrer(self):
        return Blurrer(blur_type="pixelate", blur_strength=10)

    def test_ensure_absolute_rois_relative(self, blurrer):
        boxes = [(0.1, 0.1, 0.5, 0.5)]
        shape = (100, 100)
        abs_boxes = blurrer._ensure_absolute_rois(boxes, shape)
        assert abs_boxes == [(10, 10, 40, 40)]  # x, y, w, h

    def test_ensure_absolute_rois_absolute(self, blurrer):
        boxes = [(10, 10, 50, 50)]
        shape = (100, 100)
        abs_boxes = blurrer._ensure_absolute_rois(boxes, shape)
        assert abs_boxes == [(10, 10, 40, 40)]

    def test_ensure_absolute_rois_mixed_clamped(self, blurrer):
        boxes = [(-10, -10, 110, 110)]
        shape = (100, 100)
        abs_boxes = blurrer._ensure_absolute_rois(boxes, shape)
        assert abs_boxes == [(0, 0, 100, 100)]

    def test_split_mask_rows(self, blurrer):
        rows = [
            {"mask": {"format": "binary"}, "x1": 0, "y1": 0},
            {"x1": 10, "y1": 10, "x2": 20, "y2": 20},
        ]
        shape = (100, 100)
        masks, boxes = blurrer._split_mask_rows(rows, shape, 0)
        assert len(masks) == 1
        assert len(boxes) == 1
        assert masks[0] == rows[0]
        assert boxes[0] == rows[1]

    def test_build_frame_mask_no_decoder(self, blurrer):
        # Should return empty mask if no decoder
        shape = (100, 100)
        mask = blurrer._build_frame_mask(shape, [], [])
        assert mask.shape == (100, 100)
        assert not np.any(mask)

    def test_apply_masks_to_frame(self, blurrer):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Mock _build_frame_mask to return a mask covering top-left 10x10
        with patch.object(blurrer, "_build_frame_mask") as mock_build:
            mask = np.zeros((100, 100), dtype=bool)
            mask[0:10, 0:10] = True
            mock_build.return_value = mask

            # Mock _apply_blur_to_roi to return white frame
            with patch.object(blurrer, "_apply_blur_to_roi") as mock_blur:
                mock_blur.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 255

                result = blurrer._apply_masks_to_frame(frame, [])

                # Check that only masked region is white
                assert np.all(result[0:10, 0:10] == 255)
                assert np.all(result[10:, 10:] == 0)

    def test_apply_detections_to_image(self, blurrer):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8).copy()
        detections = pl.DataFrame(
            {
                "x1": [10.0],
                "y1": [10.0],
                "x2": [20.0],
                "y2": [20.0],
                "frame": [0],
                "is_confident": [True],
            }
        )

        result = blurrer._apply_detections_to_image(img.copy(), detections)
        assert result.shape == img.shape

    def test_convert_relative_to_absolute_rois(self):
        boxes = [(0.1, 0.1, 0.5, 0.5), (0.6, 0.6, 0.9, 0.9)]
        shape = (100, 100)
        abs_boxes = convert_relative_to_absolute_rois(boxes, shape)
        assert len(abs_boxes) == 2
        assert abs_boxes[0] == (10, 10, 40, 40)
        assert abs_boxes[1] == (60, 60, 30, 30)

    def test_blur_type_getters(self, blurrer):
        available_types = blurrer.get_available_blur_types()
        assert "pixelate" in available_types
        assert "gaussian" in available_types
        assert blurrer.blur_type == "pixelate"

    def test_normalize_mask_region(self, blurrer):
        region = {"mask": np.ones((10, 10), dtype=bool), "x1": 0, "y1": 0}
        normalized = blurrer._normalize_mask_region(region)
        assert normalized is not None
        assert "mask" in normalized
        assert "x1" in normalized
        assert "y1" in normalized

    def test_group_rows_by_frame(self, blurrer):
        df = pl.DataFrame({"frame": [0, 0, 1, 2], "x1": [1.0, 2.0, 3.0, 4.0]})
        grouped = blurrer._group_rows_by_frame(df)
        assert 0 in grouped
        assert 1 in grouped
        assert 2 in grouped
        assert len(grouped[0]) == 2
        assert len(grouped[1]) == 1

    def test_blur_type_debug(self):
        blurrer = Blurrer(blur_type="debug")
        assert blurrer.blur_type == "debug"

    def test_set_blur_settings(self, blurrer):
        blurrer.set_blur_settings(blur_type="gaussian", blur_strength=15)
        assert blurrer.blur_type == "gaussian"
        assert blurrer.blur_strength == 15

    def test_ensure_absolute_rois_empty(self, blurrer):
        boxes = []
        shape = (100, 100)
        abs_boxes = blurrer._ensure_absolute_rois(boxes, shape)
        assert abs_boxes == []

    def test_apply_detections_debug_mode(self):
        blurrer = Blurrer(blur_type="debug")
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8).copy()
        detections = pl.DataFrame(
            {
                "x1": [10.0],
                "y1": [10.0],
                "x2": [20.0],
                "y2": [20.0],
                "frame": [0],
                "is_confident": [True],
            }
        )

        result = blurrer._apply_detections_to_image(img.copy(), detections)
        assert result.shape == img.shape
