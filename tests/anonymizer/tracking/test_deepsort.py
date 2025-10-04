from anonymizer.config import TrackerParams
from anonymizer.tracking.deepsort import DeepSortTracker


def test_deepsort_tracker_initializes_with_params(tmp_path):
    params = TrackerParams()
    tracker = DeepSortTracker(video_source=tmp_path / "video.mp4", params=params)
    assert tracker.params == params
