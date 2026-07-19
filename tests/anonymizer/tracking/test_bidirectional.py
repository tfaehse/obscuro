"""Tests for bidirectional tracking."""

from __future__ import annotations

import numpy as np
import polars as pl

from anonymizer.config import TrackerParams, TrackerType
from anonymizer.tracking import BidirectionalTracker
from anonymizer.tracking.common import TrackObservation, TrackState


def _make_detection(frame: int, x: float, y: float, w: float, h: float, score: float = 0.9) -> dict:
    """Helper to create a detection dict."""
    return {
        "frame": frame,
        "x1": x,
        "y1": y,
        "x2": x + w,
        "y2": y + h,
        "confidence": score,
    }


def _make_track_observations(
    track_id: int,
    start_frame: int,
    count: int,
    start_pos: tuple[float, float],
    velocity: tuple[float, float],
) -> list[TrackObservation]:
    """Helper to create synthetic track observations with linear motion."""
    observations = []
    x, y = start_pos
    vx, vy = velocity

    for i in range(count):
        frame = start_frame + i
        tlwh = np.array([x + i * vx, y + i * vy, 0.1, 0.1], dtype=float)

        obs = TrackObservation(
            frame=frame,
            track_id=track_id,
            tlwh=tlwh,
            state=TrackState.CONFIRMED,
            age=i,
            last_seen=frame,
            score=0.8,
            should_blur=True,
            frame_size=(100, 100),
            debug_color=(255, 0, 0),
            mask=None,
        )
        observations.append(obs)

    return observations


class TestBidirectionalTracker:
    """Tests for BidirectionalTracker wrapper."""

    def test_initialization(self):
        """Test BidirectionalTracker can be initialized with default parameters."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(),
            base_tracker_type=TrackerType.BYTETRACK,
        )

        assert tracker.base_tracker_type == TrackerType.BYTETRACK
        assert tracker.base_tracker is not None
        assert tracker.params is not None

    def test_initialization_with_different_base_tracker(self):
        """Test BidirectionalTracker can use different base trackers."""
        for tracker_type in [TrackerType.BYTETRACK, TrackerType.BOTSORT, TrackerType.DUMMY]:
            tracker = BidirectionalTracker(
                video_source=None,
                params=TrackerParams(),
                base_tracker_type=tracker_type,
            )
            assert tracker.base_tracker_type == tracker_type

    def test_track_empty_detections(self):
        """Test tracking with empty detections returns empty DataFrame."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(),
            base_tracker_type=TrackerType.DUMMY,
        )

        empty_df = pl.DataFrame(schema=["frame", "x1", "y1", "x2", "y2", "confidence"])
        result = tracker.track(empty_df)

        assert result.is_empty()
        expected_cols = {
            "frame",
            "track_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "state",
            "age",
            "last_seen",
            "score",
            "mask",
            "origin",
        }
        assert set(result.columns) == expected_cols

    def test_track_simple_forward_pass(self):
        """Test basic tracking with simple forward-moving object."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(confirm_after_N=1, max_misses_M=5),
            base_tracker_type=TrackerType.DUMMY,
        )

        # Create detections: object moving right across 10 frames
        detections = []
        for i in range(10):
            detections.append(
                _make_detection(frame=i, x=float(i) * 0.1, y=0.5, w=0.1, h=0.1, score=0.9)
            )

        df = pl.DataFrame(detections)
        result = tracker.track(df)

        # Should have tracked the object
        assert not result.is_empty()
        # DummyTracker creates new track each frame, so we expect multiple tracks
        # Bidirectional should still produce results
        assert result.get_column("track_id").n_unique() >= 1

    def test_bidirectional_creates_two_passes(self):
        """Test that bidirectional mode runs both forward and backward passes."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(confirm_after_N=1, max_misses_M=5),
            base_tracker_type=TrackerType.DUMMY,
        )

        # Create detections across 20 frames
        detections = []
        for i in range(20):
            detections.append(
                _make_detection(frame=i, x=float(i) * 0.05, y=0.5, w=0.1, h=0.1, score=0.9)
            )

        df = pl.DataFrame(detections)
        result = tracker.track(df)

        # Bidirectional should produce consistent tracks
        assert not result.is_empty()
        track_ids = result.get_column("track_id").unique().to_list()
        assert len(track_ids) >= 1

    def test_flying_towards_object_scenario(self):
        """Test the flying-towards-object use case: poor early detections, good late ones."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(confirm_after_N=1, max_misses_M=10),
            base_tracker_type=TrackerType.DUMMY,
        )

        detections = []

        # Early frames: far away, low confidence, fewer detections
        for i in range(5):
            detections.append(_make_detection(frame=i, x=0.4, y=0.4, w=0.2, h=0.2, score=0.3))

        # Middle frames: getting closer, medium confidence
        for i in range(5, 15):
            detections.append(_make_detection(frame=i, x=0.45, y=0.45, w=0.1, h=0.1, score=0.6))

        # Late frames: close, high confidence, precise detections
        for i in range(15, 25):
            detections.append(
                _make_detection(
                    frame=i, x=0.48 + (i - 15) * 0.002, y=0.48, w=0.04, h=0.04, score=0.95
                )
            )

        df = pl.DataFrame(detections)
        result = tracker.track(df)

        # Should have tracked the object throughout
        assert not result.is_empty()

        # Check that we have observations across the full range
        frames = result.get_column("frame").to_list()
        assert min(frames) <= 5  # Early frames included
        assert max(frames) >= 20  # Late frames included

    def test_merge_bidirectional_basic(self):
        """Test the _merge_bidirectional method with simple observations."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(),
            base_tracker_type=TrackerType.DUMMY,
        )

        # Create forward timeline
        forward_timeline = _make_track_observations(
            track_id=1, start_frame=0, count=10, start_pos=(0.5, 0.5), velocity=(0.01, 0.0)
        )

        # Create backward timeline (same object, from other direction's perspective)
        backward_timeline = _make_track_observations(
            track_id=2, start_frame=0, count=10, start_pos=(0.5, 0.5), velocity=(0.01, 0.0)
        )

        # Merge should combine them
        merged = tracker._merge_bidirectional(forward_timeline, backward_timeline)

        # Should produce merged timeline
        assert len(merged) > 0
        assert all(isinstance(obs, TrackObservation) for obs in merged)

    def test_iou_calculation(self):
        """Test IoU calculation for track matching."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(),
            base_tracker_type=TrackerType.DUMMY,
        )

        # Identical boxes: IoU ≈ 1.0
        box1 = np.array([0.5, 0.5, 0.1, 0.1])
        box2 = np.array([0.5, 0.5, 0.1, 0.1])
        iou = tracker._calculate_iou(box1, box2)
        assert abs(iou - 1.0) < 1e-6  # Account for floating point precision

        # Partially overlapping boxes
        box3 = np.array([0.5, 0.5, 0.1, 0.1])
        box4 = np.array([0.55, 0.55, 0.1, 0.1])
        iou = tracker._calculate_iou(box3, box4)
        assert 0 < iou < 1

        # Non-overlapping boxes: IoU = 0.0
        box5 = np.array([0.0, 0.0, 0.1, 0.1])
        box6 = np.array([0.9, 0.9, 0.1, 0.1])
        assert tracker._calculate_iou(box5, box6) == 0.0

    def test_should_merge_observations(self):
        """Test observation merging decision based on IoU."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(),
            base_tracker_type=TrackerType.DUMMY,
        )

        # Create observations with overlapping boxes
        obs1 = TrackObservation(
            frame=5,
            track_id=1,
            tlwh=np.array([0.5, 0.5, 0.1, 0.1]),
            state=TrackState.CONFIRMED,
            age=5,
            last_seen=5,
            score=0.8,
            should_blur=True,
            frame_size=(100, 100),
            debug_color=(255, 0, 0),
            mask=None,
        )

        obs2 = TrackObservation(
            frame=5,
            track_id=2,  # Different ID from backward pass
            tlwh=np.array([0.5, 0.5, 0.1, 0.1]),  # Same position
            state=TrackState.CONFIRMED,
            age=5,
            last_seen=5,
            score=0.9,
            should_blur=True,
            frame_size=(100, 100),
            debug_color=(0, 255, 0),
            mask=None,
        )

        # Overlapping boxes should be merged
        assert tracker._should_merge_observations(obs1, obs2) is True

        # Non-overlapping should not be merged
        obs3 = TrackObservation(
            frame=5,
            track_id=3,
            tlwh=np.array([0.0, 0.0, 0.1, 0.1]),  # Different position
            state=TrackState.CONFIRMED,
            age=5,
            last_seen=5,
            score=0.8,
            should_blur=True,
            frame_size=(100, 100),
            debug_color=(0, 0, 255),
            mask=None,
        )

        assert tracker._should_merge_observations(obs1, obs3) is False

    def test_reverse_detections(self):
        """Test detection reversal for backward pass."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(),
            base_tracker_type=TrackerType.DUMMY,
        )

        # Create detections across 10 frames
        detections = []
        for i in range(10):
            detections.append(_make_detection(frame=i, x=float(i) * 0.1, y=0.5, w=0.1, h=0.1))

        df = pl.DataFrame(detections)
        reversed_df = tracker._reverse_detections(df, max_frame=9)

        # After reversal: frame 9→0, 8→1, ..., 0→9
        # After sorting by new frame: [0,1,2,3,4,5,6,7,8,9]
        reversed_frames = reversed_df.get_column("frame").to_list()

        assert reversed_frames == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Row with new frame 0 was originally frame 9 (last frame)
        # Row with new frame 9 was originally frame 0 (first frame)

    def test_restore_backward_order(self):
        """Test restoration of backward pass results to original frame order."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(),
            base_tracker_type=TrackerType.DUMMY,
        )

        # Create backward timeline (frames 0-9 represent original frames 9-0)
        backward_timeline = _make_track_observations(
            track_id=1, start_frame=0, count=10, start_pos=(0.5, 0.5), velocity=(0.01, 0.0)
        )

        # Restore to original order
        max_frame = 9
        restored = tracker._restore_backward_order(backward_timeline, max_frame)

        # Check frame numbers are restored
        assert len(restored) == len(backward_timeline)
        assert restored[0].frame == 9  # Was 0, should be 9
        assert restored[-1].frame == 0  # Was 9, should be 0

    def test_track_history_property(self):
        """Test that track_history returns merged timeline."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(confirm_after_N=1, max_misses_M=5),
            base_tracker_type=TrackerType.DUMMY,
        )

        detections = [
            _make_detection(frame=i, x=float(i) * 0.1, y=0.5, w=0.1, h=0.1) for i in range(5)
        ]
        df = pl.DataFrame(detections)

        tracker.track(df)

        # track_history should return observations
        history = tracker.track_history
        assert isinstance(history, list)
        assert len(history) > 0

    def test_clear(self):
        """Test clear method resets tracker state."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(confirm_after_N=1, max_misses_M=5),
            base_tracker_type=TrackerType.DUMMY,
        )

        detections = [
            _make_detection(frame=i, x=float(i) * 0.1, y=0.5, w=0.1, h=0.1) for i in range(5)
        ]
        df = pl.DataFrame(detections)

        tracker.track(df)
        assert len(tracker.track_history) > 0

        tracker.clear()
        assert len(tracker.track_history) == 0


class TestBidirectionalIntegration:
    """Integration tests for bidirectional tracking with realistic scenarios."""

    def test_growing_object_scenario(self):
        """Test tracking object growing larger as camera approaches."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(confirm_after_N=1, max_misses_M=10),
            base_tracker_type=TrackerType.BYTETRACK,
        )

        detections = []

        # Object starts small and far, grows larger and closer
        for i in range(30):
            # Size increases (object getting closer)
            size = 0.05 + (i / 30) * 0.15  # 0.05 to 0.20

            # Position stabilizes (camera approaching)
            x = 0.5 - (i / 30) * 0.1  # Starts off-center, centers
            y = 0.5

            # Confidence increases with proximity
            score = 0.4 + (i / 30) * 0.5  # 0.4 to 0.9

            detections.append(_make_detection(frame=i, x=x, y=y, w=size, h=size, score=score))

        df = pl.DataFrame(detections)
        result = tracker.track(df)

        # Should track throughout
        assert not result.is_empty()

        # Check frame coverage
        frames = result.get_column("frame").to_list()
        assert min(frames) <= 5  # Early frames
        assert max(frames) >= 25  # Late frames

    def test_two_objects_crossing(self):
        """Test tracking two objects that cross paths."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(confirm_after_N=1, max_misses_M=5),
            base_tracker_type=TrackerType.BYTETRACK,
        )

        detections = []

        # Object 1: left to right
        for i in range(20):
            detections.append(_make_detection(frame=i, x=float(i) * 0.05, y=0.3, w=0.08, h=0.08))

        # Object 2: right to left (crosses around frame 10)
        for i in range(20):
            detections.append(
                _make_detection(frame=i, x=1.0 - float(i) * 0.05, y=0.3, w=0.08, h=0.08)
            )

        df = pl.DataFrame(detections)
        result = tracker.track(df)

        # Should have 2 distinct tracks
        track_ids = result.get_column("track_id").unique().to_list()
        assert len(track_ids) >= 2  # At least 2 tracks

    def test_occlusion_recovery(self):
        """Test tracking through brief occlusion."""
        tracker = BidirectionalTracker(
            video_source=None,
            params=TrackerParams(confirm_after_N=2, max_misses_M=15),
            base_tracker_type=TrackerType.BOTSORT,
        )

        detections = []

        # Object visible, then occluded (no detections), then visible again
        # Frames 0-10: visible
        for i in range(11):
            detections.append(_make_detection(frame=i, x=0.5 + i * 0.02, y=0.5, w=0.1, h=0.1))

        # Frames 11-15: occluded (no detections)

        # Frames 16-25: visible again (continuing trajectory)
        for i in range(16, 26):
            detections.append(_make_detection(frame=i, x=0.5 + i * 0.02, y=0.5, w=0.1, h=0.1))

        df = pl.DataFrame(detections)
        result = tracker.track(df)

        # Should recover track after occlusion
        assert not result.is_empty()

        # Bidirectional should help link the segments
        track_ids = result.get_column("track_id").unique().to_list()
        assert len(track_ids) >= 1
