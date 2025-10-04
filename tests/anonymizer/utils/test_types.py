"""
Tests for utility types.
"""

import numpy as np
import pytest

from anonymizer.utils.types import Detection, Track


class TestDetection:
    """Test Detection utility class."""

    def test_detection_creation(self):
        """Test creating a detection instance with x1, y1, x2, y2 format."""
        detection = Detection(x1=0.1, y1=0.2, x2=0.6, y2=0.8, confidence=0.85, object_class=1)

        assert detection.x1 == 0.1
        assert detection.y1 == 0.2
        assert detection.x2 == 0.6
        assert detection.y2 == 0.8
        assert detection.confidence == 0.85
        assert detection.object_class == 1

    def test_detection_confidence_range(self):
        """Test detection confidence is in valid range."""
        detection1 = Detection(0.1, 0.2, 0.6, 0.8, 0.0, 1)
        detection2 = Detection(0.1, 0.2, 0.6, 0.8, 1.0, 1)
        detection3 = Detection(0.1, 0.2, 0.6, 0.8, 0.5, 1)

        assert 0.0 <= detection1.confidence <= 1.0
        assert 0.0 <= detection2.confidence <= 1.0
        assert 0.0 <= detection3.confidence <= 1.0

    def test_detection_box_property(self):
        """Test the box property returns correct tuple."""
        detection = Detection(0.1, 0.2, 0.6, 0.8, 0.85, 1)
        box = detection.box

        assert box == (0.1, 0.2, 0.6, 0.8)

    def test_detection_xywh_property(self):
        """Test the xywh property returns correct format."""
        detection = Detection(0.1, 0.2, 0.6, 0.8, 0.85, 1)
        xywh = detection.xywh

        assert xywh[0] == pytest.approx(0.1)
        assert xywh[1] == pytest.approx(0.2)
        assert xywh[2] == pytest.approx(0.5)  # width
        assert xywh[3] == pytest.approx(0.6)  # height

    def test_detection_equality(self):
        """Test detection equality comparison."""
        detection1 = Detection(0.1, 0.2, 0.6, 0.8, 0.85, 1)
        detection2 = Detection(0.1, 0.2, 0.6, 0.8, 0.85, 1)
        detection3 = Detection(0.2, 0.3, 0.7, 0.9, 0.90, 2)

        assert detection1 == detection2
        assert detection1 != detection3

        # Test equality with non-Detection object (should return False)
        assert detection1 != "not a detection"
        assert detection1 != 42
        assert detection1 is not None
        assert detection1 != {"x1": 0.1, "y1": 0.2}

    def test_detection_string_representation(self):
        """Test detection string representation."""
        detection = Detection(0.1, 0.2, 0.6, 0.8, 0.85, 1)
        repr_str = repr(detection)

        assert "Detection(" in repr_str
        assert "0.85" in repr_str  # confidence should be in the string
        assert "confidence" in repr_str.lower()


class TestTrack:
    """Test Track utility class (inherits from Detection)."""

    def test_track_creation(self):
        """Test creating a track instance with track_id."""
        track = Track(0.1, 0.2, 0.6, 0.8, 0.85, 1, track_id=42)

        # Should have all Detection properties
        assert track.x1 == 0.1
        assert track.y1 == 0.2
        assert track.x2 == 0.6
        assert track.y2 == 0.8
        assert track.confidence == 0.85
        assert track.object_class == 1

        # Plus the track_id
        assert track.track_id == 42

    def test_track_inheritance(self):
        """Test that Track inherits from Detection properly."""
        track = Track(0.1, 0.2, 0.6, 0.8, 0.85, 1, track_id=42)

        # Should inherit all Detection methods
        assert hasattr(track, "box")
        assert hasattr(track, "xywh")

        # Test the inherited properties work
        assert track.box == (0.1, 0.2, 0.6, 0.8)
        assert track.xywh == pytest.approx((0.1, 0.2, 0.5, 0.6))
        assert track.object_class == 1

    def test_track_methods_inheritance(self):
        """Test that Track inherits Detection methods."""
        track1 = Track(0.1, 0.2, 0.6, 0.8, 0.85, 1, track_id=42)
        track2 = Track(0.1, 0.2, 0.6, 0.8, 0.85, 1, track_id=43)  # Different track_id

        # Should inherit equality comparison (but different track_id doesn't matter for Detection equality)
        assert track1 == track2  # Detection equality doesn't consider track_id

        # Should inherit string representation
        repr_str = repr(track1)
        assert "Detection(" in repr_str  # Uses Detection's __repr__
        assert "0.85" in repr_str

    def test_track_with_different_track_ids(self):
        """Test tracks with different track_ids."""
        track1 = Track(0.1, 0.2, 0.6, 0.8, 0.85, 1, track_id=100)
        track2 = Track(0.2, 0.3, 0.7, 0.9, 0.90, 2, track_id=200)

        assert track1.track_id == 100
        assert track2.track_id == 200
        assert track1.track_id != track2.track_id


class TestUtilityFunctions:
    """Test utility functions from the types module."""

    def test_hardware_detection_functions(self):
        """Test hardware detection utilities if available."""
        try:
            from anonymizer.utils.types import get_device_info, has_gpu

            # These may or may not exist, so we just test if they're callable
            if has_gpu:
                assert callable(has_gpu)
            if get_device_info:
                assert callable(get_device_info)
        except ImportError:
            # Hardware detection functions not implemented
            pass

    def test_coordinate_conversion_functions(self):
        """Test coordinate conversion utilities if available."""
        try:
            from anonymizer.utils.types import convert_coordinates

            if convert_coordinates:
                assert callable(convert_coordinates)
        except ImportError:
            # Coordinate conversion not implemented
            pass

    def test_center_distance_calculation(self):
        """Test relative center-distance calculation if available."""
        try:
            from anonymizer.tracking.common import center_distance

            tlwh_a = np.array([0.0, 0.0, 10.0, 10.0], dtype=float)
            tlwh_b = np.array([10.0, 0.0, 10.0, 10.0], dtype=float)
            frame_size = (100, 100)
            distance = center_distance(tlwh_a, frame_size, tlwh_b, frame_size)
            assert distance == pytest.approx(0.1)
        except ImportError:
            # Center-distance calculation not implemented
            pass

    def test_nms_function(self):
        """Test Non-Maximum Suppression if available."""
        try:
            from anonymizer.utils.types import non_max_suppression

            if non_max_suppression:
                assert callable(non_max_suppression)
        except ImportError:
            # NMS not implemented
            pass


class TestTypeValidation:
    """Test type validation functions."""

    def test_detection_validation(self):
        """Test detection validation if available."""
        try:
            from anonymizer.utils.types import is_valid_detection

            if is_valid_detection:
                valid_detection = Detection(0.1, 0.2, 0.6, 0.8, 0.85, 1)
                assert is_valid_detection(valid_detection) is True

                # Test invalid detection (negative dimensions)
                invalid_detection = Detection(0.6, 0.8, 0.1, 0.2, 0.85, 1)  # x2 < x1, y2 < y1
                assert is_valid_detection(invalid_detection) is False
        except ImportError:
            # Detection validation not implemented
            pass

    def test_coordinate_validation(self):
        """Test coordinate validation functions."""
        try:
            from anonymizer.utils.types import validate_coordinates

            if validate_coordinates:
                assert callable(validate_coordinates)
        except ImportError:
            pass

    def test_image_validation(self):
        """Test image validation functions."""
        try:
            from anonymizer.utils.types import validate_image_array

            if validate_image_array:
                # Test with valid numpy array
                image = np.ones((480, 640, 3), dtype=np.uint8)
                assert validate_image_array(image) is True

                # Test with invalid array
                invalid_image = np.ones((10,), dtype=np.uint8)
                assert validate_image_array(invalid_image) is False
        except ImportError:
            pass
