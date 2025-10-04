from __future__ import annotations

from pathlib import Path

from anonymizer.config import TrackerParams

from .base import BaseTracker


class DeepSortTracker(BaseTracker):
    """Legacy alias of the base tracker now operating without appearance cues."""

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


__all__ = ["DeepSortTracker"]
