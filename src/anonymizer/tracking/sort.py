from __future__ import annotations

from pathlib import Path

from anonymizer.config import TrackerParams

from .base import BaseTracker


class SortTracker(BaseTracker):
    """Classic SORT variant using center-distance gating with a Kalman predictor."""

    def __init__(
        self,
        video_source: str | Path | None,
        *,
        params: TrackerParams | None = None,
        cancel_event=None,
        progress_callback=None,
    ) -> None:
        super().__init__(
            video_source,
            params=params,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )


__all__ = ["SortTracker"]
