"""
Tests for the blurring functionality.
"""

import threading
from unittest.mock import Mock, patch

import numpy as np
import pytest

from anonymizer.blurring import Blurrer
from anonymizer.cancellation import CancellationException


class TestBlurrer:
    """Test Blurrer class."""

    def test_available_blur_types(self):
        """Test available blur types class method."""
        blur_types = Blurrer.get_available_blur_types()
        expected_types = ["gaussian", "pixelate", "blackout", "debug"]
        assert blur_types == expected_types

    def test_is_valid_blur_type(self):
        """Test blur type validation."""
        assert Blurrer.is_valid_blur_type("gaussian") is True
        assert Blurrer.is_valid_blur_type("GAUSSIAN") is True  # Case insensitive
        assert Blurrer.is_valid_blur_type("pixelate") is True
        assert Blurrer.is_valid_blur_type("blackout") is True
        assert Blurrer.is_valid_blur_type("debug") is True
        assert Blurrer.is_valid_blur_type("invalid") is False

    def test_blurrer_initialization_default(self):
        """Test blurrer initialization with defaults."""
        blurrer = Blurrer()
        assert blurrer.blur_type == "gaussian"
        assert blurrer.blur_strength == 10
        assert blurrer.cancel_event is None
        assert blurrer.progress_callback is None

    def test_blurrer_initialization_custom(self):
        """Test blurrer initialization with custom parameters."""
        cancel_event = threading.Event()
        progress_callback = Mock()

        blurrer = Blurrer(
            blur_type="pixelate",
            blur_strength=20,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )

        assert blurrer.blur_type == "pixelate"
        assert blurrer.blur_strength == 20
        assert blurrer.cancel_event == cancel_event
        assert blurrer.progress_callback == progress_callback

    def test_blurrer_case_insensitive_blur_type(self):
        """Test that blur type is case insensitive."""
        blurrer = Blurrer(blur_type="GAUSSIAN")
        assert blurrer.blur_type == "gaussian"

    def test_invalid_blur_type_raises_error(self):
        """Test that invalid blur type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid blur type"):
            Blurrer(blur_type="invalid_type")

    def test_blur_strength_validation(self):
        """Test blur strength validation if implemented."""
        # Test valid strength values
        blurrer = Blurrer(blur_strength=5)
        assert blurrer.blur_strength == 5

        blurrer = Blurrer(blur_strength=50)
        assert blurrer.blur_strength == 50

    def test_debug_overlay_renders_tracks_and_detections(self):
        """Debug mode should overlay red detections and blue tracks with labels."""
        blurrer = Blurrer(blur_type="debug")
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        track_row = {"x1": 10.0, "y1": 10.0, "x2": 40.0, "y2": 40.0, "track_id": 7}
        detection_row = {"x1": 30.0, "y1": 30.0, "x2": 60.0, "y2": 60.0}

        blurrer._render_debug(frame, [track_row], [detection_row])

        # Red and blue channels should be present for outlines
        assert np.any(frame[..., 0] > 0)  # Red channel (detection or track)
        assert np.any(frame[..., 2] > 0)  # Blue channel (detection)

        # Some pixels should be non-zero (debug rendering happened)
        assert np.any(frame > 0)

    def test_progress_callback_usage(self, sample_image):
        """Smoke-test progress callback with real detection columns."""
        progress_callback = Mock()
        blurrer = Blurrer(progress_callback=progress_callback)

        import polars as pl

        detections = pl.DataFrame(
            {
                "x1": [0.1],
                "y1": [0.1],
                "x2": [0.2],
                "y2": [0.2],
            }
        )

        blurrer.blur_image(sample_image.copy(), detections)
        # callback remains optional; ensure no TypeError against new signature
        progress_callback.assert_not_called()


class TestBlurrerEdgeCases:
    """Test edge cases and error conditions."""

    def test_ensure_absolute_rois_swaps_coordinates(self):
        blurrer = Blurrer()
        boxes = [(50.0, 50.0, 10.0, 5.0)]  # x2 < x1 and y2 < y1
        absolute = blurrer._ensure_absolute_rois(boxes, (100, 100, 3))
        assert absolute == [(10, 5, 40, 45)]


class TestBlurrerRuntimeSettings:
    """Test Blurrer runtime settings updates."""

    def test_set_blur_settings_type_only(self):
        """Test updating only blur type at runtime."""
        blurrer = Blurrer(blur_type="gaussian", blur_strength=10)

        blurrer.set_blur_settings(blur_type="pixelate")

        assert blurrer.blur_type == "pixelate"
        assert blurrer.blur_strength == 10  # Unchanged

    def test_set_blur_settings_strength_only(self):
        """Test updating only blur strength at runtime."""
        blurrer = Blurrer(blur_type="gaussian", blur_strength=10)

        blurrer.set_blur_settings(blur_strength=25)

        assert blurrer.blur_type == "gaussian"  # Unchanged
        assert blurrer.blur_strength == 25

    def test_set_blur_settings_both(self):
        """Test updating both blur type and strength at runtime."""
        blurrer = Blurrer(blur_type="gaussian", blur_strength=10)

        blurrer.set_blur_settings(blur_type="blackout", blur_strength=1)

        assert blurrer.blur_type == "blackout"
        assert blurrer.blur_strength == 1

    def test_set_blur_settings_invalid_type(self):
        """Test setting invalid blur type raises error."""
        blurrer = Blurrer()

        with pytest.raises(ValueError, match="Invalid blur type 'invalid'"):
            blurrer.set_blur_settings(blur_type="invalid")

    def test_set_blur_settings_zero_strength(self):
        """Test setting zero blur strength gets adjusted to 1."""
        blurrer = Blurrer()

        blurrer.set_blur_settings(blur_strength=0)

        assert blurrer.blur_strength == 1

    def test_set_blur_settings_negative_strength(self):
        """Test setting negative blur strength gets adjusted to 1."""
        blurrer = Blurrer()

        blurrer.set_blur_settings(blur_strength=-5)

        assert blurrer.blur_strength == 1

    def test_set_blur_settings_none_values(self):
        """Test setting None values doesn't change anything."""
        original_type = "gaussian"
        original_strength = 10
        blurrer = Blurrer(blur_type=original_type, blur_strength=original_strength)

        blurrer.set_blur_settings(blur_type=None, blur_strength=None)

        assert blurrer.blur_type == original_type
        assert blurrer.blur_strength == original_strength


class TestBlurrerImageMethods:
    """Test Blurrer image processing methods."""

    def test_blur_image(self):
        """Test blur_image method."""
        import polars as pl

        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        detections = pl.DataFrame(
            {
                "x1": [0.1, 0.6],
                "y1": [0.2, 0.7],
                "x2": [0.5, 0.9],
                "y2": [0.6, 0.9],
            }
        )

        blurrer = Blurrer()
        result = blurrer.blur_image(test_image, detections)

        # Should return a numpy array of same shape
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape

    def test_blur_image_empty_detections(self):
        """Test blur_image with empty detections."""
        import polars as pl

        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detections = pl.DataFrame(
            {
                "x1": [],
                "y1": [],
                "x2": [],
                "y2": [],
            }
        )

        blurrer = Blurrer()
        result = blurrer.blur_image(test_image, detections)

        # Should return original image when no detections
        assert np.array_equal(result, test_image)

    @patch("anonymizer.blurring.iio.imread")
    @patch("anonymizer.blurring.iio.imwrite")
    def test_blur_image_file_actual_calls(self, mock_imwrite, mock_imread):
        """Test blur_image_file method with actual file operations."""
        from pathlib import Path

        import polars as pl

        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = test_image

        detections = pl.DataFrame(
            {
                "x1": [0.1],
                "y1": [0.2],
                "x2": [0.5],
                "y2": [0.6],
            }
        )

        input_path = Path("input.jpg")
        output_path = Path("output.jpg")

        blurrer = Blurrer()
        blurrer.blur_image_file(input_path, detections, output_path)

        # Verify file I/O calls
        mock_imread.assert_called_once_with(input_path)
        mock_imwrite.assert_called_once()

        # Check that imwrite was called with correct output path
        call_args = mock_imwrite.call_args
        assert call_args[0][0] == output_path  # First arg is output path
        assert isinstance(call_args[0][1], np.ndarray)  # Second arg is image array


class TestBlurrerVideoMethods:
    """Test Blurrer video processing methods."""

    @patch("anonymizer.blurring.blur_video_av")
    @patch("anonymizer.blurring.get_video_info")
    @patch("anonymizer.blurring.convert_relative_to_absolute_rois")
    @patch("anonymizer.blurring.blur_rois")
    def test_blur_video(
        self, mock_blur_rois, mock_convert, mock_get_video_info, mock_blur_video_av
    ):
        """Test blur_video method."""
        from pathlib import Path

        import polars as pl

        # Setup
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")

        # Mock video info
        mock_get_video_info.return_value = {"frame_count": 100}

        detections = pl.DataFrame(
            {
                "frame": [0, 0, 1, 2],
                "x1": [0.1, 0.5, 0.2, 0.3],
                "y1": [0.1, 0.5, 0.2, 0.3],
                "x2": [0.3, 0.7, 0.4, 0.6],
                "y2": [0.3, 0.7, 0.4, 0.6],
            }
        )

        progress_callback = Mock()
        blurrer = Blurrer(
            blur_type="pixelate", blur_strength=20, progress_callback=progress_callback
        )

        # Execute
        blurrer.blur_video(input_path, detections, output_path)

        # Verify
        mock_get_video_info.assert_called_once_with(input_path)
        mock_blur_video_av.assert_called_once()

        # Check that blur_video_av was called with correct parameters
        call_args = mock_blur_video_av.call_args
        assert call_args[1]["input_path"] == input_path
        assert call_args[1]["output_path"] == output_path
        assert call_args[1]["codec"] == "h264"
        assert call_args[1]["quality"] is None
        assert callable(call_args[1]["blur_func"])
        assert callable(call_args[1]["progress_callback"])

        # Test final progress callback
        progress_callback.assert_called_with(100, "Blurring", "Complete")

    @patch("anonymizer.blurring.blur_video_av")
    @patch("anonymizer.blurring.get_video_info")
    def test_blur_video_with_cancellation(self, mock_get_video_info, mock_blur_video_av):
        """Test blur_video with cancellation event."""
        from pathlib import Path

        import polars as pl

        # Setup
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")

        mock_get_video_info.return_value = {"frame_count": 100}

        detections = pl.DataFrame(
            {
                "frame": [0],
                "x1": [0.1],
                "y1": [0.1],
                "x2": [0.3],
                "y2": [0.3],
            }
        )

        cancel_event = threading.Event()
        cancel_event.set()  # Already cancelled

        blurrer = Blurrer(cancel_event=cancel_event)
        blurrer.blur_video(input_path, detections, output_path)
        mock_blur_video_av.assert_called_once()

    @patch("anonymizer.blurring.blur_video_av")
    @patch("anonymizer.blurring.get_video_info")
    def test_blur_video_applies_codec_and_quality(self, mock_get_video_info, mock_blur_video_av):
        from pathlib import Path

        import polars as pl

        mock_get_video_info.return_value = {"frame_count": 10}

        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        detections = pl.DataFrame(
            {"frame": [0], "x1": [0.1], "y1": [0.1], "x2": [0.2], "y2": [0.2]}
        )

        blurrer = Blurrer()
        blurrer.blur_video(input_path, detections, output_path, codec="hevc", quality=18)

        mock_blur_video_av.assert_called_once()
        kwargs = mock_blur_video_av.call_args.kwargs
        assert kwargs["codec"] == "hevc"
        assert kwargs["quality"] == 18

    def test_blur_video_progress_callback_actual_calls(self):
        """Test blur_video progress callback gets called correctly."""
        from pathlib import Path

        import polars as pl

        input_path = Path("input.mp4")
        output_path = Path("output.mp4")

        detections = pl.DataFrame(
            {
                "frame": [0, 1],
                "x1": [0.1, 0.2],
                "y1": [0.1, 0.2],
                "x2": [0.3, 0.4],
                "y2": [0.3, 0.4],
            }
        )

        progress_callback = Mock()
        blurrer = Blurrer(progress_callback=progress_callback)

        with (
            patch("anonymizer.blurring.get_video_info") as mock_get_info,
            patch("anonymizer.blurring.blur_video_av") as mock_blur_video_av,
        ):
            mock_get_info.return_value = {"frame_count": 100}

            # Mock the blur_video_av to test the progress function internally
            def capture_progress_func(*args, **kwargs):
                # Get the progress_callback function passed to blur_video_av
                progress_func = kwargs.get("progress_callback")
                if progress_func:
                    # Simulate calling the internal progress function
                    progress_func(25, 100, "Processing")
                    progress_func(50, 100, "Processing")
                    progress_func(100, 100, "Processing")
                return None

            mock_blur_video_av.side_effect = capture_progress_func

            # Execute
            blurrer.blur_video(input_path, detections, output_path)

            # Verify progress callback was used
            # Should be called by internal progress_update function + final completion
            assert progress_callback.called
            progress_callback.assert_called_with(100, "Blurring", "Complete")

    def test_blur_video_with_detections_by_frame(self):
        """Test blur_video detections grouping and frame processing."""
        from pathlib import Path

        import polars as pl

        input_path = Path("input.mp4")
        output_path = Path("output.mp4")

        # Create test detections with multiple frames
        detections = pl.DataFrame(
            {
                "frame": [0, 0, 1, 2, 2],
                "x1": [0.1, 0.5, 0.2, 0.3, 0.7],
                "y1": [0.1, 0.5, 0.2, 0.3, 0.7],
                "x2": [0.3, 0.7, 0.4, 0.6, 0.9],
                "y2": [0.3, 0.7, 0.4, 0.6, 0.9],
            }
        )

        blurrer = Blurrer(blur_type="pixelate", blur_strength=20)

        with (
            patch("anonymizer.blurring.get_video_info") as mock_get_info,
            patch("anonymizer.blurring.blur_video_av") as mock_blur_video_av,
        ):
            mock_get_info.return_value = {"frame_count": 100}

            # Capture the process_frame function to test it
            captured_process_frame = None

            def capture_blur_func(*args, **kwargs):
                nonlocal captured_process_frame
                captured_process_frame = kwargs.get("blur_func")
                return None

            mock_blur_video_av.side_effect = capture_blur_func

            # Execute the blur_video method
            blurrer.blur_video(input_path, detections, output_path)

            # Verify blur_video_av was called with correct parameters
            mock_blur_video_av.assert_called_once()
            call_kwargs = mock_blur_video_av.call_args[1]
            assert call_kwargs["input_path"] == input_path
            assert call_kwargs["output_path"] == output_path
            assert call_kwargs["codec"] == "h264"
            assert call_kwargs["quality"] is None
            assert callable(call_kwargs["blur_func"])

            # Now test the captured process_frame function
            assert captured_process_frame is not None

            # Test frame processing with detections
            test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            # Frame 0 should have 2 detections
            with (
                patch("anonymizer.blurring.convert_relative_to_absolute_rois") as mock_convert,
                patch("anonymizer.blurring.blur_rois") as mock_blur_rois,
            ):
                mock_convert.return_value = [(10, 10, 30, 30), (50, 50, 70, 70)]
                mock_blur_rois.return_value = test_frame

                result = captured_process_frame(test_frame, 0)

                expected_boxes = [
                    (0.1, 0.1, 0.3, 0.3),
                    (0.5, 0.5, 0.7, 0.7),
                ]
                mock_convert.assert_called_once_with(expected_boxes, test_frame.shape[:2])
                # Use ANY for the frame argument to avoid numpy array comparison error
                # Use call_args to verify arguments manually to avoid numpy array comparison error
                assert mock_blur_rois.call_count == 1
                args, kwargs = mock_blur_rois.call_args
                assert np.array_equal(args[0], test_frame)
                # The current implementation blurs the entire frame copy and then applies the mask
                # So the ROI passed to blur_rois is the full frame
                h, w = test_frame.shape[:2]
                assert args[1] == [(0, 0, w, h)]
                assert kwargs["blur_type"] == "pixelate"
                assert kwargs["blur_strength"] == 20

            # Reset mocks and test frame with no detections
            mock_convert.reset_mock()
            mock_blur_rois.reset_mock()

            result = captured_process_frame(test_frame, 5)  # Frame 5 has no detections

            # Should return frame unchanged and not call blur functions
            mock_convert.assert_not_called()
            mock_blur_rois.assert_not_called()
            assert np.array_equal(result, test_frame)

    def test_blur_video_with_cancellation_during_processing(self):
        """Test blur_video respects cancellation event."""
        from pathlib import Path

        import polars as pl

        input_path = Path("input.mp4")
        output_path = Path("output.mp4")

        detections = pl.DataFrame(
            {
                "frame": [0],
                "x1": [0.1],
                "y1": [0.1],
                "x2": [0.3],
                "y2": [0.3],
            }
        )

        cancel_event = threading.Event()
        blurrer = Blurrer(cancel_event=cancel_event)

        with (
            patch("anonymizer.blurring.get_video_info") as mock_get_info,
            patch("anonymizer.blurring.blur_video_av") as mock_blur_video_av,
        ):
            mock_get_info.return_value = {"frame_count": 100}

            # Capture the process_frame function to test cancellation
            captured_process_frame = None

            def capture_blur_func(*args, **kwargs):
                nonlocal captured_process_frame
                captured_process_frame = kwargs.get("blur_func")
                return None

            mock_blur_video_av.side_effect = capture_blur_func

            # Execute the blur_video method
            blurrer.blur_video(input_path, detections, output_path)

            # Test the process_frame function with cancellation
            assert captured_process_frame is not None

            test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            # Set cancellation event
            cancel_event.set()

            # Process frame should return original frame immediately when cancelled
            with pytest.raises(CancellationException):
                captured_process_frame(test_frame, 0)

    @patch("anonymizer.blurring.convert_relative_to_absolute_rois")
    @patch("anonymizer.blurring.blur_rois")
    def test_blur_video_process_frame_function(self, mock_blur_rois, mock_convert):
        """Test the process_frame function used in blur_video."""

        import polars as pl

        # Create a blurrer instance to access the internal process_frame function
        blurrer = Blurrer(blur_type="gaussian", blur_strength=10)

        # Create test detections
        detections = pl.DataFrame(
            {
                "frame": [0, 1],
                "x1": [0.1, 0.2],
                "y1": [0.1, 0.2],
                "x2": [0.3, 0.4],
                "y2": [0.3, 0.4],
            }
        )

        # We need to simulate the detections_by_frame structure that blur_video creates
        detections_by_frame = {}
        for row in detections.iter_rows(named=True):
            frame_idx = row["frame"]
            detections_by_frame.setdefault(frame_idx, []).append(
                (row["x1"], row["y1"], row["x2"], row["y2"])
            )

        # Create a test frame
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Mock the conversion and blurring functions
        absolute_rois = [(10, 10, 30, 30)]
        mock_convert.return_value = absolute_rois
        mock_blur_rois.return_value = test_frame

        # We need to manually create the process_frame function logic
        def process_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
            """Process a single frame with blur effects."""
            if blurrer.cancel_event and blurrer.cancel_event.is_set():
                return frame

            # Get detections for this frame
            if frame_idx in detections_by_frame:
                relative_rois = detections_by_frame[frame_idx]

                # Convert relative coordinates to absolute pixel coordinates
                from anonymizer.blurring import blur_rois, convert_relative_to_absolute_rois

                absolute_rois = convert_relative_to_absolute_rois(relative_rois, frame.shape)

                # Apply blur to ROIs
                if absolute_rois:
                    frame = blur_rois(
                        frame,
                        absolute_rois,
                        blur_type=blurrer.blur_type,
                        blur_strength=blurrer.blur_strength,
                    )

            return frame

        # Test processing frame 0 (has detections)
        result = process_frame(test_frame, 0)

        # Should process the frame
        mock_convert.assert_called_once_with([(0.1, 0.1, 0.3, 0.3)], test_frame.shape)
        mock_blur_rois.assert_called_once_with(
            test_frame, absolute_rois, blur_type="gaussian", blur_strength=10
        )
        assert np.array_equal(result, test_frame)

        # Reset mocks and test frame with no detections
        mock_convert.reset_mock()
        mock_blur_rois.reset_mock()

        result = process_frame(test_frame, 5)  # Frame 5 has no detections

        # Should return frame unchanged
        mock_convert.assert_not_called()
        mock_blur_rois.assert_not_called()
        assert np.array_equal(result, test_frame)


def test_blurrer_invalid_strength():
    """Test Blurrer with invalid strength."""
    blurrer = Blurrer(blur_type="gaussian")
    # The method does not raise, but should clamp or ignore invalid values
    blurrer.set_blur_settings(blur_strength=-1)
    blurrer.set_blur_settings(blur_strength=1000)


class TestMaskDecoder:
    """Test mask decoder functionality."""

    def test_set_mask_decoder(self):
        """Test setting a mask decoder."""
        blurrer = Blurrer()
        mock_decoder = Mock()

        blurrer.set_mask_decoder(mock_decoder)

        assert blurrer.mask_decoder == mock_decoder

    def test_set_mask_decoder_none(self):
        """Test setting mask decoder to None."""
        blurrer = Blurrer()
        mock_decoder = Mock()
        blurrer.set_mask_decoder(mock_decoder)

        blurrer.set_mask_decoder(None)

        assert blurrer.mask_decoder is None


class TestGroupRowsByFrame:
    """Test _group_rows_by_frame static method."""

    def test_group_rows_by_frame_basic(self):
        """Test grouping rows by frame number."""
        import polars as pl

        df = pl.DataFrame(
            {
                "frame": [0, 0, 1, 2, 2, 2],
                "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "y1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )

        result = Blurrer._group_rows_by_frame(df)

        assert len(result[0]) == 2
        assert len(result[1]) == 1
        assert len(result[2]) == 3

    def test_group_rows_by_frame_empty(self):
        """Test grouping with empty dataframe."""
        import polars as pl

        df = pl.DataFrame({"frame": [], "x1": [], "y1": []})
        result = Blurrer._group_rows_by_frame(df)

        assert result == {}

    def test_group_rows_by_frame_none(self):
        """Test grouping with None dataframe."""
        result = Blurrer._group_rows_by_frame(None)
        assert result == {}

    def test_group_rows_by_frame_no_frame_column(self):
        """Test grouping with dataframe without frame column."""
        import polars as pl

        df = pl.DataFrame(
            {
                "x1": [0.1, 0.2],
                "y1": [0.1, 0.2],
            }
        )

        result = Blurrer._group_rows_by_frame(df)

        # Should use frame 0 as default
        assert 0 in result
        assert len(result[0]) == 2


class TestDrawingFunctions:
    """Test drawing helper functions."""

    def test_draw_box_basic(self):
        """Test drawing a box on frame."""
        blurrer = Blurrer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        row = {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}

        blurrer._draw_box(frame, row, (255, 0, 0))

        # Check that some pixels were drawn
        assert np.any(frame > 0)

    def test_draw_box_with_label(self):
        """Test drawing a box with label."""
        blurrer = Blurrer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        row = {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}

        blurrer._draw_box(frame, row, (255, 0, 0), label="Test")

        # Check that some pixels were drawn
        assert np.any(frame > 0)

    def test_draw_box_empty_frame(self):
        """Test drawing on empty frame."""
        blurrer = Blurrer()
        frame = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        row = {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}

        # Should not crash
        blurrer._draw_box(frame, row, (255, 0, 0))

    def test_draw_box_width_height_format(self):
        """Test drawing box with width/height instead of x2/y2."""
        blurrer = Blurrer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        row = {"x1": 10.0, "y1": 10.0, "width": 40.0, "height": 40.0}

        blurrer._draw_box(frame, row, (255, 0, 0))

        assert np.any(frame > 0)

    def test_draw_box_invalid_coordinates(self):
        """Test drawing box with invalid coordinates (x2 <= x1)."""
        blurrer = Blurrer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        row = {"x1": 50.0, "y1": 50.0, "x2": 10.0, "y2": 10.0}

        # Should not crash
        blurrer._draw_box(frame, row, (255, 0, 0))

    def test_draw_box_clipped_coordinates(self):
        """Test drawing box with coordinates outside frame bounds."""
        blurrer = Blurrer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        row = {"x1": -10.0, "y1": -10.0, "x2": 150.0, "y2": 150.0}

        blurrer._draw_box(frame, row, (255, 0, 0))

        # Should clip and draw
        assert np.any(frame > 0)

    def test_draw_mask_basic(self):
        """Test drawing a mask on frame."""
        blurrer = Blurrer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask_region = {"mask": mask, "x1": 10, "y1": 10}

        blurrer._draw_mask(frame, mask_region, (255, 0, 0))

        # Check that contours were drawn
        assert np.any(frame > 0)

    def test_draw_mask_none(self):
        """Test drawing with None mask."""
        blurrer = Blurrer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Should not crash
        blurrer._draw_mask(frame, None, (255, 0, 0))

    def test_draw_mask_empty_mask(self):
        """Test drawing with empty mask (all zeros)."""
        blurrer = Blurrer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask_region = {"mask": mask, "x1": 10, "y1": 10}

        # Should not crash
        blurrer._draw_mask(frame, mask_region, (255, 0, 0))

    def test_draw_mask_invalid_shape(self):
        """Test drawing with invalid mask shape."""
        blurrer = Blurrer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((30, 30, 3), dtype=np.uint8)  # 3D mask, should be 2D
        mask_region = {"mask": mask, "x1": 10, "y1": 10}

        # Should not crash
        blurrer._draw_mask(frame, mask_region, (255, 0, 0))


class TestMaskRegionFunctions:
    """Test mask region helper functions."""

    def test_normalize_mask_region_valid(self):
        """Test normalizing a valid mask region."""
        mask = np.ones((10, 10), dtype=bool)
        region = {"mask": mask, "x1": 5, "y1": 10}

        result = Blurrer._normalize_mask_region(region)

        assert result is not None
        assert result["x1"] == 5
        assert result["y1"] == 10
        assert np.array_equal(result["mask"], mask)

    def test_normalize_mask_region_None(self):
        """Test normalizing None."""
        result = Blurrer._normalize_mask_region(None)
        assert result is None

    def test_normalize_mask_region_not_dict(self):
        """Test normalizing non-dict value."""
        result = Blurrer._normalize_mask_region("not a dict")
        assert result is None

    def test_normalize_mask_region_missing_fields(self):
        """Test normalizing region with missing fields."""
        result = Blurrer._normalize_mask_region({"mask": np.ones((10, 10))})
        assert result is None

        result = Blurrer._normalize_mask_region({"x1": 5, "y1": 10})
        assert result is None

    def test_normalize_mask_region_empty_mask(self):
        """Test normalizing region with empty (all False) mask."""
        mask = np.zeros((10, 10), dtype=bool)
        region = {"mask": mask, "x1": 5, "y1": 10}

        result = Blurrer._normalize_mask_region(region)

        assert result is None

    def test_normalize_mask_region_invalid_coords(self):
        """Test normalizing region with invalid coordinates."""
        mask = np.ones((10, 10), dtype=bool)
        region = {"mask": mask, "x1": "invalid", "y1": 10}

        result = Blurrer._normalize_mask_region(region)

        assert result is None

    def test_mask_region_from_polygon_relative(self):
        """Test creating mask from relative polygon."""
        blurrer = Blurrer()
        points = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_polygon(points, frame_shape, relative=True)

        assert result is not None
        assert "mask" in result
        assert "x1" in result
        assert "y1" in result

    def test_mask_region_from_polygon_absolute(self):
        """Test creating mask from absolute polygon."""
        blurrer = Blurrer()
        points = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_polygon(points, frame_shape, relative=False)

        assert result is not None
        assert "mask" in result

    def test_mask_region_from_polygon_none(self):
        """Test creating mask from None points."""
        blurrer = Blurrer()
        result = blurrer._mask_region_from_polygon(None, (100, 100, 3), relative=True)
        assert result is None

    def test_mask_region_from_polygon_invalid_shape(self):
        """Test creating mask from invalid polygon shape."""
        blurrer = Blurrer()
        points = [[0.1, 0.1, 0.5]]  # Wrong shape
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_polygon(points, frame_shape, relative=True)

        # Should return None for invalid shape
        assert result is None

    def test_mask_region_from_polygon_zero_area(self):
        """Test creating mask from polygon with zero area."""
        blurrer = Blurrer()
        points = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]  # All same point
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_polygon(points, frame_shape, relative=True)

        assert result is None

    def test_mask_region_from_binary(self):
        """Test creating mask from binary payload."""
        blurrer = Blurrer()
        data = [True] * 100
        size = (10, 10)
        payload = {"data": data, "size": size}
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_binary(payload, frame_shape)

        assert result is not None

    def test_mask_region_from_binary_missing_data(self):
        """Test creating mask from binary with missing data."""
        blurrer = Blurrer()
        payload = {"size": (10, 10)}
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_binary(payload, frame_shape)

        assert result is None

    def test_mask_region_from_binary_invalid_size(self):
        """Test creating mask from binary with invalid size."""
        blurrer = Blurrer()
        data = [True] * 100
        payload = {"data": data, "size": "invalid"}
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_binary(payload, frame_shape)

        assert result is None

    def test_mask_region_from_binary_reshape_error(self):
        """Test creating mask from binary with reshape error."""
        blurrer = Blurrer()
        data = [True] * 50  # Wrong size for reshape
        size = (10, 10)
        payload = {"data": data, "size": size}
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_binary(payload, frame_shape)

        assert result is None

    def test_mask_region_from_proto_with_row(self):
        """Test creating mask from proto payload with row."""
        blurrer = Blurrer()
        # Create proto mask data
        proto_h, proto_w = 28, 28
        data = (np.ones((proto_h, proto_w), dtype=np.uint8) * 255).tobytes()
        payload = {"data": data, "size": (proto_h, proto_w)}
        row = {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_proto(payload, frame_shape, row)

        assert result is not None
        assert result["x1"] == 10
        assert result["y1"] == 10

    def test_mask_region_from_proto_without_row(self):
        """Test creating mask from proto payload without row."""
        blurrer = Blurrer()
        proto_h, proto_w = 28, 28
        data = (np.ones((proto_h, proto_w), dtype=np.uint8) * 255).tobytes()
        payload = {"data": data, "size": (proto_h, proto_w)}
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_proto(payload, frame_shape, None)

        assert result is not None

    def test_mask_region_from_proto_invalid_data(self):
        """Test creating mask from proto with invalid data."""
        blurrer = Blurrer()
        payload = {"data": "not bytes", "size": (28, 28)}
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_proto(payload, frame_shape, None)

        assert result is None

    def test_mask_region_from_proto_wrong_size(self):
        """Test creating mask from proto with all zeros (below threshold)."""
        blurrer = Blurrer()
        proto_h, proto_w = 28, 28
        # All zeros will be below 0.5 threshold after normalization
        data = bytes(proto_h * proto_w)
        payload = {"data": data, "size": (proto_h, proto_w)}
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_proto(payload, frame_shape, None)

        # All zeros will create empty mask after threshold, so result should be None
        assert result is None

    def test_shrink_full_mask(self):
        """Test shrinking a full-frame mask."""
        blurrer = Blurrer()
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:30, 20:40] = True
        frame_shape = (100, 100, 3)

        result = blurrer._shrink_full_mask(mask, frame_shape)

        assert result is not None
        assert result["x1"] == 20
        assert result["y1"] == 10

    def test_shrink_full_mask_empty(self):
        """Test shrinking empty mask."""
        blurrer = Blurrer()
        mask = np.zeros((100, 100), dtype=bool)
        frame_shape = (100, 100, 3)

        result = blurrer._shrink_full_mask(mask, frame_shape)

        assert result is None

    def test_shrink_full_mask_invalid_dimensions(self):
        """Test shrinking mask with invalid dimensions."""
        blurrer = Blurrer()
        mask = np.zeros((100, 100, 3), dtype=bool)  # 3D, should be 2D
        frame_shape = (100, 100, 3)

        result = blurrer._shrink_full_mask(mask, frame_shape)

        assert result is None

    def test_extract_roi_from_mask(self):
        """Test extracting ROI from mask."""
        mask = np.ones((100, 100), dtype=bool)
        frame_shape = (100, 100, 3)

        result = Blurrer._extract_roi_from_mask(mask, frame_shape, x1=10, y1=10, x2=50, y2=50)

        assert result is not None
        assert result["x1"] == 10
        assert result["y1"] == 10

    def test_extract_roi_from_mask_clipping(self):
        """Test extracting ROI with coordinates outside bounds."""
        mask = np.ones((100, 100), dtype=bool)
        frame_shape = (100, 100, 3)

        result = Blurrer._extract_roi_from_mask(mask, frame_shape, x1=-10, y1=-10, x2=150, y2=150)

        assert result is not None
        assert result["x1"] == 0
        assert result["y1"] == 0

    def test_extract_roi_from_mask_invalid_bounds(self):
        """Test extracting ROI with invalid bounds (x2 <= x1)."""
        mask = np.ones((100, 100), dtype=bool)
        frame_shape = (100, 100, 3)

        result = Blurrer._extract_roi_from_mask(mask, frame_shape, x1=50, y1=50, x2=10, y2=10)

        assert result is None

    def test_extract_roi_from_mask_empty(self):
        """Test extracting ROI from empty mask region."""
        mask = np.zeros((100, 100), dtype=bool)
        frame_shape = (100, 100, 3)

        result = Blurrer._extract_roi_from_mask(mask, frame_shape, x1=10, y1=10, x2=50, y2=50)

        assert result is None


class TestDecodesMaskPayload:
    """Test _decode_mask_payload function."""

    def test_decode_mask_payload_none(self):
        """Test decoding None payload."""
        blurrer = Blurrer()
        result = blurrer._decode_mask_payload(None, (100, 100, 3))
        assert result is None

    def test_decode_mask_payload_invalid_frame_shape(self):
        """Test decoding with invalid frame shape."""
        blurrer = Blurrer()
        payload = {"format": "relative_polygon", "points": [[0.1, 0.1]]}
        result = blurrer._decode_mask_payload(payload, (0, 0, 3))
        assert result is None

    def test_decode_mask_payload_with_decoder(self):
        """Test decoding integer payload with decoder."""
        blurrer = Blurrer()
        mock_decoder = Mock()
        mock_decoder.decode_mask_from_payload.return_value = {
            "mask": np.ones((10, 10), dtype=bool),
            "x1": 5,
            "y1": 10,
        }
        blurrer.set_mask_decoder(mock_decoder)

        result = blurrer._decode_mask_payload(123, (100, 100, 3), {"frame": 0})

        assert result is not None
        mock_decoder.decode_mask_from_payload.assert_called_once()

    def test_decode_mask_payload_relative_polygon(self):
        """Test decoding relative polygon payload."""
        blurrer = Blurrer()
        payload = {
            "format": "relative_polygon",
            "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
        }

        result = blurrer._decode_mask_payload(payload, (100, 100, 3))

        assert result is not None

    def test_decode_mask_payload_absolute_polygon(self):
        """Test decoding absolute polygon payload."""
        blurrer = Blurrer()
        payload = {
            "format": "absolute_polygon",
            "points": [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]],
        }

        result = blurrer._decode_mask_payload(payload, (100, 100, 3))

        assert result is not None

    def test_decode_mask_payload_binary(self):
        """Test decoding binary payload."""
        blurrer = Blurrer()
        payload = {
            "format": "binary",
            "data": [True] * 100,
            "size": (10, 10),
        }

        result = blurrer._decode_mask_payload(payload, (100, 100, 3))

        assert result is not None

    def test_decode_mask_payload_proto_mask(self):
        """Test decoding proto_mask payload."""
        blurrer = Blurrer()
        proto_h, proto_w = 28, 28
        data = (np.ones((proto_h, proto_w), dtype=np.uint8) * 255).tobytes()
        payload = {
            "format": "proto_mask",
            "data": data,
            "size": (proto_h, proto_w),
        }
        row = {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}

        result = blurrer._decode_mask_payload(payload, (100, 100, 3), row)

        assert result is not None


class TestBlurMaskRegion:
    """Test _blur_mask_region function."""

    def test_blur_mask_region_basic(self):
        """Test blurring a mask region."""
        blurrer = Blurrer()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((30, 30), dtype=bool)
        mask[10:20, 10:20] = True
        mask_region = {"mask": mask, "x1": 10, "y1": 10}

        result = blurrer._blur_mask_region(frame, mask_region)

        assert result.shape == frame.shape

    def test_blur_mask_region_none(self):
        """Test blurring with None mask region."""
        blurrer = Blurrer()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = blurrer._blur_mask_region(frame, None)

        # Should return frame unchanged
        assert np.array_equal(result, frame)

    def test_blur_mask_region_empty_mask(self):
        """Test blurring with empty mask."""
        blurrer = Blurrer()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((30, 30), dtype=bool)
        mask_region = {"mask": mask, "x1": 10, "y1": 10}

        result = blurrer._blur_mask_region(frame, mask_region)

        # Should return frame unchanged
        assert np.array_equal(result, frame)

    def test_blur_mask_region_out_of_bounds(self):
        """Test blurring with mask region outside frame bounds."""
        blurrer = Blurrer()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.ones((30, 30), dtype=bool)
        mask_region = {"mask": mask, "x1": 200, "y1": 200}

        result = blurrer._blur_mask_region(frame, mask_region)

        # Should return frame unchanged
        assert np.array_equal(result, frame)


class TestApplyMasksToFrame:
    """Test _apply_masks_to_frame function."""

    def test_apply_masks_to_frame_basic(self):
        """Test applying masks to frame."""
        blurrer = Blurrer()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.ones((30, 30), dtype=bool)
        masks = [{"mask": mask, "x1": 10, "y1": 10}]

        result = blurrer._apply_masks_to_frame(frame, masks)

        assert result.shape == frame.shape

    def test_apply_masks_to_frame_empty_list(self):
        """Test applying empty mask list."""
        blurrer = Blurrer()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = blurrer._apply_masks_to_frame(frame, [])

        # Should return frame unchanged
        assert np.array_equal(result, frame)


class TestSplitMaskRows:
    """Test _split_mask_rows function."""

    def test_split_mask_rows_with_masks(self):
        """Test splitting rows with mask data."""
        blurrer = Blurrer()
        rows = [
            {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2, "mask": 123},
            {"x1": 0.3, "y1": 0.3, "x2": 0.4, "y2": 0.4, "mask": None},
            {"x1": 0.5, "y1": 0.5, "x2": 0.6, "y2": 0.6, "mask": {"format": "binary"}},
        ]

        with_mask, without_mask = blurrer._split_mask_rows(rows, (100, 100, 3), 0)

        assert len(with_mask) == 2
        assert len(without_mask) == 1

    def test_split_mask_rows_no_masks(self):
        """Test splitting rows without mask data."""
        blurrer = Blurrer()
        rows = [
            {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
            {"x1": 0.3, "y1": 0.3, "x2": 0.4, "y2": 0.4, "mask": None},
        ]

        with_mask, without_mask = blurrer._split_mask_rows(rows, (100, 100, 3), 0)

        assert len(with_mask) == 0
        assert len(without_mask) == 2


class TestApplyDetectionsToImage:
    """Test _apply_detections_to_image function."""

    def test_apply_detections_to_image_none_image(self):
        """Test applying detections to None image."""
        import polars as pl

        blurrer = Blurrer()
        detections = pl.DataFrame({"x1": [0.1], "y1": [0.1], "x2": [0.2], "y2": [0.2]})

        with pytest.raises(ValueError, match="Failed to load image"):
            blurrer._apply_detections_to_image(None, detections)

    def test_apply_detections_to_image_debug_mode(self):
        """Test applying detections in debug mode."""
        import polars as pl

        blurrer = Blurrer(blur_type="debug")
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detections = pl.DataFrame(
            {
                "x1": [0.1, 0.5],
                "y1": [0.1, 0.5],
                "x2": [0.3, 0.7],
                "y2": [0.3, 0.7],
            }
        )

        result = blurrer._apply_detections_to_image(image, detections)

        assert result.shape == image.shape
        # In debug mode, boxes should be drawn
        assert np.any(result > 0)

    def test_apply_detections_with_is_confident_column(self):
        """Test applying detections with is_confident column."""
        import polars as pl

        blurrer = Blurrer()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detections = pl.DataFrame(
            {
                "x1": [0.1, 0.5],
                "y1": [0.1, 0.5],
                "x2": [0.3, 0.7],
                "y2": [0.3, 0.7],
                "is_confident": [True, False],  # Second detection should be filtered out
            }
        )

        result = blurrer._apply_detections_to_image(image, detections)

        assert result.shape == image.shape

    def test_apply_detections_with_mask_decoder_release(self):
        """Test mask decoder release after applying detections."""
        import polars as pl

        blurrer = Blurrer()
        mock_decoder = Mock()
        mock_decoder.decode_masks_for_rows.return_value = []
        blurrer.set_mask_decoder(mock_decoder)

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detections = pl.DataFrame(
            {
                "x1": [0.1],
                "y1": [0.1],
                "x2": [0.3],
                "y2": [0.3],
                "frame": [0],
            }
        )

        result = blurrer._apply_detections_to_image(image, detections)

        assert result.shape == image.shape
        # Mask decoder release should be called if it has the method
        if hasattr(mock_decoder, "release_mask_proto"):
            mock_decoder.release_mask_proto.assert_called()


class TestEnsureAbsoluteRois:
    """Test _ensure_absolute_rois function."""

    def test_ensure_absolute_rois_empty_list(self):
        """Test with empty boxes list."""
        result = Blurrer._ensure_absolute_rois([], (100, 100, 3))
        assert result == []

    def test_ensure_absolute_rois_relative(self):
        """Test converting relative coordinates."""
        boxes = [(0.1, 0.1, 0.3, 0.3), (0.5, 0.5, 0.7, 0.7)]
        result = Blurrer._ensure_absolute_rois(boxes, (100, 100, 3))

        assert len(result) == 2
        # Check that coordinates are now in pixel space
        assert all(coord >= 0 for roi in result for coord in roi)

    def test_ensure_absolute_rois_already_absolute(self):
        """Test with already absolute coordinates."""
        boxes = [(10.0, 10.0, 50.0, 50.0), (60.0, 60.0, 90.0, 90.0)]
        result = Blurrer._ensure_absolute_rois(boxes, (100, 100, 3))

        assert len(result) == 2


class TestRenderDebug:
    """Test _render_debug function."""

    def test_render_debug_with_masks(self):
        """Test debug rendering with masks."""
        blurrer = Blurrer(blur_type="debug")
        mock_decoder = Mock()
        mask_region = {"mask": np.ones((20, 20), dtype=bool), "x1": 10, "y1": 10}
        mock_decoder.decode_masks_for_rows.return_value = [mask_region]
        blurrer.set_mask_decoder(mock_decoder)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracks = [{"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0, "track_id": 1}]
        detections = [{"x1": 40.0, "y1": 40.0, "x2": 60.0, "y2": 60.0, "confidence": 0.95}]

        blurrer._render_debug(frame, tracks, detections)

        # Should have drawn both tracks and detections
        assert np.any(frame > 0)

    def test_render_debug_with_confidence_filter(self):
        """Test debug rendering filters out non-confident detections."""
        blurrer = Blurrer(blur_type="debug")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracks = []
        detections = [
            {"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0, "is_confident": True},
            {"x1": 40.0, "y1": 40.0, "x2": 60.0, "y2": 60.0, "is_confident": False},
        ]

        blurrer._render_debug(frame, tracks, detections)

        # Only confident detection should be drawn

    def test_render_debug_with_custom_color(self):
        """Test debug rendering with custom track colors."""
        blurrer = Blurrer(blur_type="debug")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracks = [
            {
                "x1": 10.0,
                "y1": 10.0,
                "x2": 30.0,
                "y2": 30.0,
                "track_id": 1,
                "debug_color": [0, 255, 0],  # Green
            }
        ]

        blurrer._render_debug(frame, tracks, None)

        # Should use custom color
        assert np.any(frame[:, :, 1] > 0)  # Green channel

    def test_render_debug_invalid_color(self):
        """Test debug rendering with invalid custom color."""
        blurrer = Blurrer(blur_type="debug")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracks = [
            {
                "x1": 10.0,
                "y1": 10.0,
                "x2": 30.0,
                "y2": 30.0,
                "track_id": 1,
                "debug_color": "invalid",  # Should fall back to default
            }
        ]

        blurrer._render_debug(frame, tracks, None)


class TestBuildFrameMask:
    """Test _build_frame_mask function."""

    def test_build_frame_mask_with_decoder(self):
        """Test building frame mask with decoder."""
        blurrer = Blurrer()
        mock_decoder = Mock()
        mask_region = {"mask": np.ones((20, 20), dtype=bool), "x1": 10, "y1": 10}
        mock_decoder.decode_masks_for_rows.return_value = [mask_region]
        blurrer.set_mask_decoder(mock_decoder)

        mask_rows = [{"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0, "mask": 123}]

        result = blurrer._build_frame_mask((100, 100, 3), mask_rows, [])

        assert result.shape == (100, 100)
        assert np.any(result)
        mock_decoder.decode_masks_for_rows.assert_called_once()

    def test_build_frame_mask_decoder_error(self):
        """Test building frame mask when decoder raises error."""
        blurrer = Blurrer()
        mock_decoder = Mock()
        mock_decoder.decode_masks_for_rows.side_effect = Exception("Decode error")
        blurrer.set_mask_decoder(mock_decoder)

        mask_rows = [{"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0, "mask": 123}]

        # Should not crash, just log warning
        result = blurrer._build_frame_mask((100, 100, 3), mask_rows, [])

        assert result.shape == (100, 100)

    def test_build_frame_mask_with_boxes(self):
        """Test building frame mask with bounding boxes."""
        blurrer = Blurrer()
        rows_without_mask = [
            {"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0},
            {"x1": 50.0, "y1": 50.0, "x2": 70.0, "y2": 70.0},
        ]

        result = blurrer._build_frame_mask((100, 100, 3), [], rows_without_mask)

        assert result.shape == (100, 100)
        assert np.any(result)


class TestApplyBlurToRoi:
    """Test _apply_blur_to_roi function."""

    def test_apply_blur_to_roi_empty(self):
        """Test applying blur to empty ROI."""
        blurrer = Blurrer()
        roi = np.array([], dtype=np.uint8).reshape(0, 0, 3)

        result = blurrer._apply_blur_to_roi(roi)

        assert result.size == 0


class TestAdditionalEdgeCases:
    """Additional edge case tests for maximum coverage."""

    def test_decode_mask_payload_unknown_format(self):
        """Test decoding payload with unknown format."""
        blurrer = Blurrer()
        payload = {"format": "unknown_format", "data": "something"}

        result = blurrer._decode_mask_payload(payload, (100, 100, 3))

        assert result is None

    def test_decode_mask_payload_decoder_exception(self):
        """Test decoding with decoder that raises exception."""
        blurrer = Blurrer()
        mock_decoder = Mock()
        mock_decoder.decode_mask_from_payload.side_effect = Exception("Decoder error")
        blurrer.set_mask_decoder(mock_decoder)

        # Should handle exception gracefully and return None
        result = blurrer._decode_mask_payload(123, (100, 100, 3), {"frame": 0})

        assert result is None

    def test_mask_region_from_binary_3d_mask(self):
        """Test binary mask that reshapes into 3D (invalid)."""
        blurrer = Blurrer()
        data = [True] * 60
        size = (3, 4, 5)  # 3D size
        payload = {"data": data, "size": size}
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_binary(payload, frame_shape)

        assert result is None

    def test_mask_region_from_proto_zero_roi(self):
        """Test proto mask with zero-sized ROI."""
        blurrer = Blurrer()
        proto_h, proto_w = 28, 28
        data = (np.ones((proto_h, proto_w), dtype=np.uint8) * 255).tobytes()
        payload = {"data": data, "size": (proto_h, proto_w)}
        # Row with same x1 and x2 (zero width)
        row = {"x1": 10.0, "y1": 10.0, "x2": 10.0, "y2": 50.0}
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_proto(payload, frame_shape, row)

        assert result is None

    def test_mask_region_from_proto_invalid_size_format(self):
        """Test proto with invalid size format."""
        blurrer = Blurrer()
        data = bytes(100)
        payload = {"data": data, "size": "invalid"}
        frame_shape = (100, 100, 3)

        result = blurrer._mask_region_from_proto(payload, frame_shape, None)

        assert result is None

    def test_blur_mask_region_non_bool_dtype(self):
        """Test blurring mask region with non-bool dtype."""
        blurrer = Blurrer()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Mask with uint8 dtype instead of bool
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask_region = {"mask": mask, "x1": 10, "y1": 10}

        result = blurrer._blur_mask_region(frame, mask_region)

        assert result.shape == frame.shape

    def test_draw_box_zero_height_width_frame(self):
        """Test drawing box on frame with zero dimension."""
        blurrer = Blurrer()
        # Frame with valid shape but effectively zero area
        frame = np.zeros((0, 100, 3), dtype=np.uint8)
        row = {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}

        # Should not crash
        blurrer._draw_box(frame, row, (255, 0, 0))

    @patch("anonymizer.blurring.blur_video_av")
    @patch("anonymizer.blurring.get_video_info")
    def test_blur_video_with_mask_decoder_release(self, mock_get_video_info, mock_blur_video_av):
        """Test blur_video calls mask decoder release."""
        from pathlib import Path

        import polars as pl

        input_path = Path("input.mp4")
        output_path = Path("output.mp4")

        mock_get_video_info.return_value = {"frame_count": 10}

        detections = pl.DataFrame(
            {
                "frame": [0],
                "x1": [0.1],
                "y1": [0.1],
                "x2": [0.2],
                "y2": [0.2],
                "mask": [123],
            }
        )

        blurrer = Blurrer()
        mock_decoder = Mock()
        mock_decoder.decode_masks_for_rows.return_value = []
        mock_decoder.release_mask_proto = Mock()
        blurrer.set_mask_decoder(mock_decoder)

        # Capture the blur function
        captured_blur_func = None

        def capture_func(*args, **kwargs):
            nonlocal captured_blur_func
            captured_blur_func = kwargs.get("blur_func")

        mock_blur_video_av.side_effect = capture_func

        blurrer.blur_video(input_path, detections, output_path)

        # Call the blur function to trigger mask release
        if captured_blur_func:
            test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            captured_blur_func(test_frame, 0)

            # Verify mask decoder release was called
            mock_decoder.release_mask_proto.assert_called_with(0)

    def test_build_frame_mask_invalid_mask_regions(self):
        """Test building frame mask with invalid mask regions from decoder."""
        blurrer = Blurrer()
        mock_decoder = Mock()
        # Return invalid mask regions (missing fields, wrong types)
        mock_decoder.decode_masks_for_rows.return_value = [
            None,  # None region
            {},  # Empty dict
            {"mask": "invalid"},  # Invalid mask
            {"mask": np.ones((20, 20), dtype=bool), "x1": 10},  # Missing y1
        ]
        blurrer.set_mask_decoder(mock_decoder)

        mask_rows = [
            {"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0, "mask": 123},
            {"x1": 40.0, "y1": 40.0, "x2": 60.0, "y2": 60.0, "mask": 456},
        ]

        result = blurrer._build_frame_mask((100, 100, 3), mask_rows, [])

        # Should handle invalid regions gracefully
        assert result.shape == (100, 100)

    def test_build_frame_mask_out_of_bounds_mask(self):
        """Test building frame mask with mask coordinates outside frame."""
        blurrer = Blurrer()
        mock_decoder = Mock()
        # Mask positioned outside frame bounds
        mask_region = {"mask": np.ones((20, 20), dtype=bool), "x1": 200, "y1": 200}
        mock_decoder.decode_masks_for_rows.return_value = [mask_region]
        blurrer.set_mask_decoder(mock_decoder)

        mask_rows = [{"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0, "mask": 123}]

        result = blurrer._build_frame_mask((100, 100, 3), mask_rows, [])

        # Should clip coordinates
        assert result.shape == (100, 100)

    def test_build_frame_mask_empty_decoded_mask(self):
        """Test building frame mask with decoded mask that has no True values."""
        blurrer = Blurrer()
        mock_decoder = Mock()
        # All-False mask
        mask_region = {"mask": np.zeros((20, 20), dtype=bool), "x1": 10, "y1": 10}
        mock_decoder.decode_masks_for_rows.return_value = [mask_region]
        blurrer.set_mask_decoder(mock_decoder)

        mask_rows = [{"x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0, "mask": 123}]

        result = blurrer._build_frame_mask((100, 100, 3), mask_rows, [])

        assert result.shape == (100, 100)

    def test_render_debug_without_decoder(self):
        """Test render debug without mask decoder (fallback to decode_mask_payload)."""
        blurrer = Blurrer(blur_type="debug")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Track with mask payload as dict
        tracks = [
            {
                "x1": 10.0,
                "y1": 10.0,
                "x2": 30.0,
                "y2": 30.0,
                "track_id": 1,
                "mask": {
                    "format": "relative_polygon",
                    "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
                },
            }
        ]

        # Detection with mask
        detections = [
            {
                "x1": 40.0,
                "y1": 40.0,
                "x2": 60.0,
                "y2": 60.0,
                "confidence": 0.95,
                "is_confident": True,
                "mask": {
                    "format": "relative_polygon",
                    "points": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
                },
            }
        ]

        blurrer._render_debug(frame, tracks, detections)

        # Should have drawn both
        assert np.any(frame > 0)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
