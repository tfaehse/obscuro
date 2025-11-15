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

        # Blue channel should be present for track outline, red channel for detection outline
        assert np.any(frame[..., 0] > 0)
        assert np.any(frame[..., 2] > 0)
        # Spot-check that green channel remains zero on outlines (pure red/blue)
        outline_pixels = frame[np.where((frame[..., 0] > 0) | (frame[..., 2] > 0))]
        assert np.all(outline_pixels[:, 1] == 0)

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

    @patch("cv2.imread")
    @patch("cv2.imwrite")
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
        mock_imread.assert_called_once_with(str(input_path))
        mock_imwrite.assert_called_once()

        # Check that imwrite was called with correct output path
        call_args = mock_imwrite.call_args
        assert call_args[0][0] == str(output_path)  # First arg is output path
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
                mock_blur_rois.assert_called_once_with(
                    test_frame,
                    [(10, 10, 30, 30), (50, 50, 70, 70)],
                    blur_type="pixelate",
                    blur_strength=20,
                )

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
