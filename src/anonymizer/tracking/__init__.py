from __future__ import annotations

from typing import Any

from anonymizer.config import TrackerParams

from .botsort import BotSortTracker
from .bytetrack import ByteTrackTracker
from .dummy import DummyTracker
from .hybrid import HybridSOTTracker
from .offline_linker import link_tracklets

TRACKER_FACTORY = {
    "dummy": DummyTracker,
    "bytetrack": ByteTrackTracker,
    "botsort": BotSortTracker,
    "hybrid_sot": HybridSOTTracker,
}


def _normalise_tracker_name(name: Any) -> str:
    raw = getattr(name, "value", name)
    if isinstance(raw, str):
        return raw.lower()
    raise ValueError(f"Unsupported tracker identifier: {name!r}")


def get_tracker(name: Any):
    resolved = _normalise_tracker_name(name)
    if resolved not in TRACKER_FACTORY:
        raise ValueError(f"Unknown tracker: {name}")
    return TRACKER_FACTORY[resolved]


class TrackerFactory:
    @staticmethod
    def get(
        name: Any,
        video_source=None,
        *,
        params: TrackerParams | dict | None = None,
        cancel_event=None,
        progress_callback=None,
    ):
        tracker_cls = get_tracker(name)
        tracker_params = params
        if tracker_params is None:
            tracker_params = TrackerParams()
        elif isinstance(tracker_params, dict):
            tracker_params = TrackerParams(**tracker_params)
        return tracker_cls(
            video_source,
            params=tracker_params,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )


__all__ = [
    "TRACKER_FACTORY",
    "TrackerFactory",
    "link_tracklets",
]
