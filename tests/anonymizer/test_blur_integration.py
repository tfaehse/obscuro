"""
Integration tests for blurring with and without masks.

These tests verify that the full pipeline works correctly:
- Detections with masks disabled should blur bounding boxes
- Detections with masks enabled should blur using masks
"""

import numpy as np
import polars as pl

from anonymizer.blurring import Blurrer


def create_mock_detections(
    frame: int = 0,
    boxes: list[tuple[float, float, float, float]] | None = None,
    masks: list[int | None] | None = None,
    frame_shape: tuple[int, int] = (100, 100),
) -> pl.DataFrame:
    """
    Create mock detection DataFrame.

    Coordinates are expected to be in absolute pixels.
    """
    if boxes is None:
        boxes = [(10, 10, 30, 30)]  # Single default box

    height, width = frame_shape
    n = len(boxes)

    mask_values = masks if masks is not None else [None] * n

    return pl.DataFrame(
        {
            "frame": [frame] * n,
            "x1": [b[0] for b in boxes],
            "y1": [b[1] for b in boxes],
            "x2": [b[2] for b in boxes],
            "y2": [b[3] for b in boxes],
            "confidence": [0.9] * n,
            "class_name": ["person"] * n,
            "object_class": [0] * n,
            "frame_width": [width] * n,
            "frame_height": [height] * n,
            "is_confident": [True] * n,
            "mask": mask_values,
        },
        schema={
            "frame": pl.Int64,
            "x1": pl.Float64,
            "y1": pl.Float64,
            "x2": pl.Float64,
            "y2": pl.Float64,
            "confidence": pl.Float64,
            "class_name": pl.String,
            "object_class": pl.Int64,
            "frame_width": pl.Int64,
            "frame_height": pl.Int64,
            "is_confident": pl.Boolean,
            "mask": pl.Int64,
        },
    )


class TestBlurIntegration:
    """Integration tests for blurring with various configurations."""

    def test_blur_with_boxes_only_no_masks(self):
        """Test that bounding boxes are blurred when masks are None."""
        # Create a simple image with a distinct region
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:30, 10:30] = 255  # White square in top-left

        # Create detections without masks (mask=None)
        detections = create_mock_detections(
            frame=0,
            boxes=[(10, 10, 30, 30)],
            masks=[None],
            frame_shape=(100, 100),
        )

        blurrer = Blurrer(blur_type="gaussian", blur_strength=10)
        result = blurrer.blur_image(image, detections)

        # The region should be blurred (not pure white anymore)
        blurred_region = result[10:30, 10:30]
        original_region = np.full((20, 20, 3), 255, dtype=np.uint8)

        # Blurred region should differ from original
        assert not np.array_equal(blurred_region, original_region), (
            "Bounding box region should be blurred"
        )

    def test_blur_with_boxes_only_two_frames(self):
        """Test that bounding boxes are blurred across multiple frames."""
        for frame_idx in range(2):
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            # Different position per frame
            y_offset = frame_idx * 40
            image[10 + y_offset : 30 + y_offset, 10:30] = 255

            detections = create_mock_detections(
                frame=frame_idx,
                boxes=[(10, 10 + y_offset, 30, 30 + y_offset)],
                masks=[None],
                frame_shape=(100, 100),
            )

            blurrer = Blurrer(blur_type="gaussian", blur_strength=10)
            result = blurrer.blur_image(image, detections)

            # The region should be blurred
            blurred_region = result[10 + y_offset : 30 + y_offset, 10:30]
            original_region = np.full((20, 20, 3), 255, dtype=np.uint8)

            assert not np.array_equal(blurred_region, original_region), (
                f"Frame {frame_idx}: Bounding box region should be blurred"
            )

    def test_blur_with_blackout(self):
        """Test that blackout mode works with bounding boxes."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Gray image

        detections = create_mock_detections(
            frame=0,
            boxes=[(20, 20, 60, 60)],
            masks=[None],
            frame_shape=(100, 100),
        )

        blurrer = Blurrer(blur_type="blackout")
        result = blurrer.blur_image(image, detections)

        # The region should be black
        assert np.all(result[20:60, 20:60] == 0), "Blackout should make the region completely black"

    def test_blur_with_pixelate(self):
        """Test that pixelate mode works with bounding boxes."""
        # Create an image with varying colors (gradient) so pixelation is visible
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(40):
            for j in range(40):
                image[20 + i, 20 + j] = [i * 6, j * 6, (i + j) * 3]

        detections = create_mock_detections(
            frame=0,
            boxes=[(20, 20, 60, 60)],
            masks=[None],
            frame_shape=(100, 100),
        )

        # Store original for comparison
        original_region = image[20:60, 20:60].copy()

        blurrer = Blurrer(blur_type="pixelate", blur_strength=10)
        result = blurrer.blur_image(image, detections)

        # The region should be modified (pixelated)
        blurred_region = result[20:60, 20:60]

        assert not np.array_equal(blurred_region, original_region), (
            "Pixelate should modify the region with gradient"
        )

    def test_blur_multiple_boxes(self):
        """Test blurring multiple bounding boxes."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:30, 10:30] = 255  # Box 1
        image[60:80, 60:80] = 128  # Box 2

        detections = create_mock_detections(
            frame=0,
            boxes=[
                (10, 10, 30, 30),
                (60, 60, 80, 80),
            ],
            masks=[None, None],
            frame_shape=(100, 100),
        )

        blurrer = Blurrer(blur_type="blackout")
        result = blurrer.blur_image(image, detections)

        # Both regions should be black
        assert np.all(result[10:30, 10:30] == 0), "First box should be blacked out"
        assert np.all(result[60:80, 60:80] == 0), "Second box should be blacked out"

    def test_no_detections_no_change(self):
        """Test that image is unchanged when there are no detections."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:30, 10:30] = 255

        # Empty detections
        detections = pl.DataFrame(
            {
                "frame": [],
                "x1": [],
                "y1": [],
                "x2": [],
                "y2": [],
                "confidence": [],
                "class_name": [],
                "object_class": [],
                "frame_width": [],
                "frame_height": [],
                "is_confident": [],
                "mask": [],
            },
            schema={
                "frame": pl.Int64,
                "x1": pl.Float64,
                "y1": pl.Float64,
                "x2": pl.Float64,
                "y2": pl.Float64,
                "confidence": pl.Float64,
                "class_name": pl.String,
                "object_class": pl.Int64,
                "frame_width": pl.Int64,
                "frame_height": pl.Int64,
                "is_confident": pl.Boolean,
                "mask": pl.Int64,
            },
        )

        blurrer = Blurrer(blur_type="blackout")
        result = blurrer.blur_image(image.copy(), detections)

        # Image should be unchanged
        assert np.array_equal(result, image), "Image should be unchanged with no detections"

    def test_debug_mode_with_boxes(self):
        """Test that debug mode renders boxes correctly."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        detections = create_mock_detections(
            frame=0,
            boxes=[(20, 20, 60, 60)],
            masks=[None],
            frame_shape=(100, 100),
        )

        blurrer = Blurrer(blur_type="debug")
        result = blurrer.blur_image(image, detections)

        # Debug mode should draw something (colored outlines)
        # The image should not be all zeros anymore
        assert np.any(result > 0), "Debug mode should draw something"


class TestBlurWithRelativeCoordinates:
    """Test blurring with relative coordinates (0-1 range)."""

    def test_blur_relative_coordinates(self):
        """Test that relative coordinates are handled correctly."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:50, 25:50] = 255  # White square in center

        # Coordinates as relative (0-1)
        detections = pl.DataFrame(
            {
                "frame": [0],
                "x1": [0.25],
                "y1": [0.25],
                "x2": [0.5],
                "y2": [0.5],
                "confidence": [0.9],
                "class_name": ["person"],
                "object_class": [0],
                "frame_width": [100],
                "frame_height": [100],
                "is_confident": [True],
                "mask": [None],
            },
            schema={
                "frame": pl.Int64,
                "x1": pl.Float64,
                "y1": pl.Float64,
                "x2": pl.Float64,
                "y2": pl.Float64,
                "confidence": pl.Float64,
                "class_name": pl.String,
                "object_class": pl.Int64,
                "frame_width": pl.Int64,
                "frame_height": pl.Int64,
                "is_confident": pl.Boolean,
                "mask": pl.Int64,
            },
        )

        blurrer = Blurrer(blur_type="blackout")
        result = blurrer.blur_image(image, detections)

        # The region (25:50, 25:50) should be black
        assert np.all(result[25:50, 25:50] == 0), "Relative coordinates should work"


class TestSplitMaskRows:
    """Test the _split_mask_rows method directly."""

    def test_split_with_none_masks(self):
        """Test that rows with None masks go to without_mask list."""
        blurrer = Blurrer()

        rows = [
            {"x1": 10, "y1": 10, "x2": 30, "y2": 30, "mask": None},
            {"x1": 40, "y1": 40, "x2": 60, "y2": 60, "mask": None},
        ]

        with_mask, without_mask = blurrer._split_mask_rows(rows, (100, 100), 0)

        assert len(with_mask) == 0, "No rows should have mask data"
        assert len(without_mask) == 2, "Both rows should be in without_mask"

    def test_split_with_int_masks(self):
        """Test that rows with int mask IDs go to with_mask list."""
        blurrer = Blurrer()

        rows = [
            {"x1": 10, "y1": 10, "x2": 30, "y2": 30, "mask": 0},
            {"x1": 40, "y1": 40, "x2": 60, "y2": 60, "mask": 1},
        ]

        with_mask, without_mask = blurrer._split_mask_rows(rows, (100, 100), 0)

        assert len(with_mask) == 2, "Both rows should have mask data"
        assert len(without_mask) == 0, "No rows should be in without_mask"

    def test_split_mixed(self):
        """Test mixed rows with and without masks."""
        blurrer = Blurrer()

        rows = [
            {"x1": 10, "y1": 10, "x2": 30, "y2": 30, "mask": 0},
            {"x1": 40, "y1": 40, "x2": 60, "y2": 60, "mask": None},
        ]

        with_mask, without_mask = blurrer._split_mask_rows(rows, (100, 100), 0)

        assert len(with_mask) == 1, "One row should have mask data"
        assert len(without_mask) == 1, "One row should be in without_mask"
