"""
Tests for Detection and Track helper classes.
"""

import pytest

from anonymizer.utils.types import Detection, Track


class TestDetection:
    def test_detection_attributes(self):
        det = Detection(0.1, 0.2, 0.6, 0.8, 0.9, 1)
        assert det.x1 == pytest.approx(0.1)
        assert det.y1 == pytest.approx(0.2)
        assert det.x2 == pytest.approx(0.6)
        assert det.y2 == pytest.approx(0.8)
        assert det.confidence == pytest.approx(0.9)
        assert det.object_class == 1

    def test_detection_box_helpers(self):
        det = Detection(0.1, 0.2, 0.6, 0.8, 0.5, 0)
        assert det.box == (0.1, 0.2, 0.6, 0.8)
        xywh = det.xywh
        assert xywh[0] == pytest.approx(0.1)
        assert xywh[1] == pytest.approx(0.2)
        assert xywh[2] == pytest.approx(0.5)
        assert xywh[3] == pytest.approx(0.6)

    def test_detection_equality(self):
        det1 = Detection(0.1, 0.2, 0.6, 0.8, 0.9, 1)
        det2 = Detection(0.1, 0.2, 0.6, 0.8, 0.9, 1)
        det3 = Detection(0.2, 0.3, 0.7, 0.9, 0.9, 2)
        assert det1 == det2
        assert det1 != det3


class TestTrack:
    def test_track_inherits_detection(self):
        track = Track(0.1, 0.2, 0.6, 0.8, 0.9, 1, track_id=42)
        assert track.track_id == 42
        assert track.box == (0.1, 0.2, 0.6, 0.8)
        assert track.xywh == pytest.approx((0.1, 0.2, 0.5, 0.6))

    def test_track_equality_ignores_track_id(self):
        track1 = Track(0.1, 0.2, 0.6, 0.8, 0.9, 1, track_id=1)
        track2 = Track(0.1, 0.2, 0.6, 0.8, 0.9, 1, track_id=2)
        assert track1 == track2
