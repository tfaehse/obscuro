import numpy as np
import polars as pl
import pytest

from anonymizer.config import TrackerParams
from anonymizer.tracking.botsort import BotSortTracker
from anonymizer.tracking.bytetrack import ByteTrackTracker
from anonymizer.tracking.common import (
    Detection,
    TrackObservation,
    TrackState,
    batched_center_distance,
)
from anonymizer.tracking.dummy import DummyTracker
from anonymizer.tracking.hybrid import HybridSOTTracker
from anonymizer.tracking.offline_linker import link_tracklets
from anonymizer.tracking.sort import SortTracker


def test_dummy_tracker_smooths_and_dilates_boxes():
    params = TrackerParams(ema_alpha=0.5, bbox_dilate_pct=0.1)
    tracker = DummyTracker(None, params=params)
    data = pl.DataFrame(
        {
            "frame": [0, 1],
            "x1": [10.0, 11.0],
            "y1": [10.0, 11.0],
            "x2": [50.0, 51.0],
            "y2": [50.0, 51.0],
            "confidence": [0.9, 0.9],
            "frame_width": [640, 640],
            "frame_height": [480, 480],
        }
    )

    tracks = tracker.track(data)
    assert tracks["track_id"].n_unique() == 1

    first = tracks.filter(pl.col("frame") == 0).row(0, named=True)
    second = tracks.filter(pl.col("frame") == 1).row(0, named=True)

    assert first["x2"] - first["x1"] == pytest.approx(44.0)
    assert second["x2"] - second["x1"] == pytest.approx(44.0)
    assert second["x1"] == pytest.approx(8.5)


def test_dummy_tracker_reconfigure_and_empty_input():
    tracker = DummyTracker(None)
    detections = pl.DataFrame(
        {
            "frame": [0],
            "x1": [10.0],
            "y1": [10.0],
            "x2": [20.0],
            "y2": [20.0],
            "confidence": [0.9],
            "frame_width": [100],
            "frame_height": [100],
        }
    )
    tracker.track(detections)
    assert tracker._ema_state

    tracker.reconfigure(TrackerParams(ema_alpha=0.7))
    assert tracker._ema_state == {}
    assert tracker._track_ids == {}
    tracker_info = tracker.get_tracker_info()
    assert tracker_info["tracker_type"] == "DummyTracker"

    empty_df = pl.DataFrame(
        schema={
            "frame": pl.Int64,
            "x1": pl.Float64,
            "y1": pl.Float64,
            "x2": pl.Float64,
            "y2": pl.Float64,
        }
    )
    empty_result = tracker.track(empty_df)
    assert empty_result.is_empty()


def test_dummy_tracker_prune_state_removes_stale_slots():
    tracker = DummyTracker(None)
    tracker._ema_state = {0: np.array([1, 1, 1, 1]), 3: np.array([2, 2, 2, 2])}
    tracker._track_ids = {0: 1, 3: 2}
    tracker._prune_state(1)
    assert list(tracker._ema_state.keys()) == [0]
    np.testing.assert_array_equal(tracker._ema_state[0], np.array([1, 1, 1, 1]))
    assert tracker._track_ids == {0: 1}


def test_bytetrack_low_score_pool_bridges_gap():
    tracker = ByteTrackTracker(None, params=TrackerParams(use_low_score_pool=True))
    data = pl.DataFrame(
        {
            "frame": [0, 1, 2],
            "x1": [0.0, 0.0, 0.0],
            "y1": [0.0, 0.0, 0.0],
            "x2": [40.0, 40.0, 40.0],
            "y2": [40.0, 40.0, 40.0],
            "confidence": [0.9, 0.25, 0.9],
            "frame_width": [640, 640, 640],
            "frame_height": [480, 480, 480],
        }
    )

    tracks = tracker.track(data)
    assert tracks["track_id"].n_unique() == 1
    history_frames = [obs.frame for obs in tracker.track_history if obs.track_id == 1]
    assert history_frames == [0, 1, 2]


def test_botsort_stage_two_matches_without_appearance():
    params = TrackerParams(
        distance_gate_hi=0.1,
        distance_gate_lo=0.5,
        cam_motion_comp=False,
        use_low_score_pool=False,
    )
    tracker = BotSortTracker(None, params=params)
    data = pl.DataFrame(
        {
            "frame": [0, 1],
            "x1": [0.0, 20.0],
            "y1": [0.0, 0.0],
            "x2": [20.0, 40.0],
            "y2": [20.0, 20.0],
            "confidence": [0.9, 0.9],
            "frame_width": [100, 100],
            "frame_height": [100, 100],
            "is_confident": [True, True],
        }
    )

    tracker.track(data)
    history_frames = [obs.frame for obs in tracker.track_history if obs.track_id == 1]
    assert history_frames == [0, 1]


def test_botsort_motion_compensation_updates_prev_centers():
    params = TrackerParams(
        distance_gate_hi=0.2,
        distance_gate_lo=0.3,
        cam_motion_comp=True,
    )
    tracker = BotSortTracker(None, params=params)
    data = pl.DataFrame(
        {
            "frame": [0, 1],
            "x1": [0.0, 0.0],
            "y1": [0.0, 0.0],
            "x2": [20.0, 20.0],
            "y2": [20.0, 20.0],
            "confidence": [0.9, 0.9],
            "frame_width": [100, 100],
            "frame_height": [100, 100],
            "is_confident": [True, True],
        }
    )

    tracker.track(data)
    assert tracker._prev_centers == [(10.0, 10.0)]


def test_botsort_reconfigure_clears_prev_centers():
    tracker = BotSortTracker(None, params=TrackerParams())
    tracker._prev_centers = [(5.0, 5.0)]
    tracker.reconfigure(TrackerParams())
    assert tracker._prev_centers == []


def test_sort_tracker_basic_association():
    tracker = SortTracker(None)
    data = pl.DataFrame(
        {
            "frame": [0, 0],
            "x1": [0.0, 100.0],
            "y1": [0.0, 100.0],
            "x2": [50.0, 150.0],
            "y2": [50.0, 150.0],
            "confidence": [0.9, 0.9],
            "frame_width": [640, 640],
            "frame_height": [480, 480],
        }
    )

    tracker.track(data)
    history_ids = {obs.track_id for obs in tracker.track_history}
    assert history_ids == {1, 2}


def test_hybrid_visual_tracker_updates(monkeypatch):
    params = TrackerParams(use_visual_tracker=True, vt_max_age=3)
    tracker = HybridSOTTracker(None, params=params)

    class FakeTracker:
        def __init__(self):
            self.initialised = False

        def init(self, frame, bbox):
            self.initialised = True
            self.bbox = bbox

        def update(self, frame):
            return True, self.bbox

    monkeypatch.setattr(
        "anonymizer.tracking.hybrid._create_visual_tracker",
        lambda backend: FakeTracker(),
    )
    monkeypatch.setattr(tracker, "_get_frame", lambda idx: np.zeros((10, 10, 3), dtype=np.uint8))

    det_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    detection = Detection(
        frame_idx=0,
        tlwh=det_array.copy(),
        score=0.9,
        frame_size=(10, 10),
    )

    tracker._start_new_track(detection, 0)
    track_obj = tracker._tracks[0]

    tracker._update_track(track_obj, detection, 0)
    assert track_obj.visual_tracker is not None
    tracker._mark_missed(track_obj, 1)
    assert track_obj.misses == 0


def _make_track(
    track_id: int,
    start_frame: int,
    length: int,
    start_pos: tuple[float, float],
    velocity: tuple[float, float],
) -> list[TrackObservation]:
    observations: list[TrackObservation] = []
    width, height = 20.0, 16.0
    base_age = 1
    for i in range(length):
        frame = start_frame + i
        cx = start_pos[0] + velocity[0] * i
        cy = start_pos[1] + velocity[1] * i
        tlwh = np.array([cx - width / 2.0, cy - height / 2.0, width, height], dtype=float)
        observations.append(
            TrackObservation(
                frame=frame,
                track_id=track_id,
                tlwh=tlwh,
                state=TrackState.CONFIRMED,
                age=base_age + i,
                last_seen=frame,
                score=0.9,
                should_blur=True,
                frame_size=(1920, 1080),
            )
        )
    return observations


def test_offline_linker_merges_tracklets():
    segment_a = _make_track(
        11, start_frame=0, length=12, start_pos=(100.0, 200.0), velocity=(3.0, 0.5)
    )
    segment_b = _make_track(
        22,
        start_frame=15,
        length=12,
        start_pos=(100.0 + 3.0 * 12, 200.0 + 0.5 * 12),
        velocity=(3.1, 0.6),
    )

    params = TrackerParams(distance_gate=0.4, offline_linker_max_misses=30)
    mapping, filled = link_tracklets("video", [*segment_a, *segment_b], params)
    assert mapping[11] == mapping[22]

    merged_id = mapping[11]
    frames = [obs.frame for obs in filled[merged_id]]
    # gap of 3 frames should now be filled
    assert frames[0] == segment_a[0].frame
    assert frames[-1] == segment_b[-1].frame
    assert set(range(segment_a[-1].frame + 1, segment_b[0].frame)).issubset(frames)


def test_offline_linker_respects_frame_gap():
    segment_a = _make_track(31, start_frame=0, length=12, start_pos=(0.0, 0.0), velocity=(1.0, 0.0))
    segment_b = _make_track(
        32, start_frame=50, length=12, start_pos=(12.0, 0.0), velocity=(1.0, 0.0)
    )

    params = TrackerParams(offline_linker_max_misses=5)
    mapping, _ = link_tracklets("video", [*segment_a, *segment_b], params)
    assert mapping[31] != mapping[32]


def test_offline_linker_uses_custom_max_gap():
    segment_a = _make_track(41, start_frame=0, length=12, start_pos=(0.0, 0.0), velocity=(1.0, 0.5))
    segment_b = _make_track(
        42, start_frame=25, length=12, start_pos=(12.0, 6.0), velocity=(1.05, 0.55)
    )

    params = TrackerParams(offline_linker_max_misses=30)
    mapping, _ = link_tracklets("video", [*segment_a, *segment_b], params)
    assert mapping[41] == mapping[42]


def test_offline_linker_rejects_mismatched_direction():
    params = TrackerParams(distance_gate=0.5, offline_linker_max_misses=30)

    segment_a = _make_track(51, start_frame=0, length=12, start_pos=(0.0, 0.0), velocity=(2.0, 0.0))
    segment_b = _make_track(
        52, start_frame=15, length=12, start_pos=(0.0, 0.0), velocity=(-2.0, 0.0)
    )

    mapping, _ = link_tracklets("video", [*segment_a, *segment_b], params)
    assert mapping[51] != mapping[52]


def test_offline_linker_returns_interpolated_track():
    params = TrackerParams(distance_gate=0.5, offline_linker_max_misses=30)
    segment_a = _make_track(
        61, start_frame=0, length=12, start_pos=(50.0, 60.0), velocity=(1.5, 0.8)
    )
    segment_b = _make_track(
        62,
        start_frame=14,
        length=12,
        start_pos=(50.0 + 1.5 * 12, 60.0 + 0.8 * 12),
        velocity=(1.6, 0.82),
    )

    mapping, filled = link_tracklets("video", [*segment_a, *segment_b], params)
    assert mapping[61] == mapping[62]
    new_id = mapping[61]
    track = filled[new_id]
    expected_start = segment_a[0].frame
    expected_end = segment_b[-1].frame
    frames = [obs.frame for obs in track]
    assert frames[0] == expected_start
    assert frames[-1] == expected_end
    gap_frames = set(range(segment_a[-1].frame + 1, segment_b[0].frame))
    assert gap_frames.issubset(frames)
    gap_examples = [frame for frame in frames if frame in gap_frames]
    for frame in gap_examples:
        idx = frames.index(frame)
        assert track[idx].should_blur is False


def test_batched_center_distance_handles_empty_inputs():
    result = batched_center_distance([], [], [], [])
    assert result.shape == (0, 0)
