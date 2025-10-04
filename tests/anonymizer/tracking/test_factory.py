import pytest

from anonymizer.config import TrackerType
from anonymizer.tracking import TRACKER_FACTORY, TrackerFactory, get_tracker


def test_get_tracker_by_string():
    tracker_cls = get_tracker("dummy")
    assert tracker_cls is TRACKER_FACTORY["dummy"]


def test_get_tracker_by_enum():
    tracker_cls = get_tracker(TrackerType.BOTSORT)
    assert tracker_cls is TRACKER_FACTORY["botsort"]


def test_get_tracker_unknown():
    with pytest.raises(ValueError):
        get_tracker("unknown")


def test_get_tracker_invalid_identifier_type():
    with pytest.raises(ValueError):
        get_tracker(123)


def test_tracker_factory_accepts_dict_params():
    tracker = TrackerFactory.get("bytetrack", params={"distance_gate": 0.55})
    assert tracker.params.distance_gate == 0.55
