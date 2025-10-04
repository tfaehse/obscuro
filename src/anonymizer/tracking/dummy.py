from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from anonymizer.config import TrackerParams

from .common import TrackObservation, TrackState


class DummyTracker:
    """Tracker that simply echoes detections with optional EMA smoothing and dilation."""

    def __init__(
        self,
        video_source: str | Path | None,
        *,
        params: TrackerParams | None = None,
        cancel_event=None,
        progress_callback=None,
    ) -> None:
        self.video_source = Path(video_source) if video_source else None
        self.params = params or TrackerParams()
        self.cancel_event = cancel_event
        self.progress_callback = progress_callback

        self._ema_state: dict[int, np.ndarray] = {}
        self._track_ids: dict[int, int] = {}
        self._next_track_id = 1
        # Maintain a simple per-frame observation timeline to satisfy the
        # interface used by components that introspect track_history.
        self._timeline: list[TrackObservation] = []

    def set_video_source(self, source: Path | str | None) -> None:
        self.video_source = Path(source) if source else None

    def reconfigure(self, params: TrackerParams) -> None:
        self.params = params
        # Reset EMA state when params change to avoid mixing weights
        self._ema_state.clear()
        self._track_ids.clear()
        self._next_track_id = 1

    def track(self, detections: pl.DataFrame) -> pl.DataFrame:
        if detections.is_empty():
            self._ema_state.clear()
            self._track_ids.clear()
            self._timeline.clear()
            return pl.DataFrame(
                schema={
                    "frame": pl.Int64,
                    "track_id": pl.Int64,
                    "x1": pl.Float64,
                    "y1": pl.Float64,
                    "x2": pl.Float64,
                    "y2": pl.Float64,
                    "width": pl.Float64,
                    "height": pl.Float64,
                    "state": pl.Utf8,
                    "age": pl.Int64,
                    "last_seen": pl.Int64,
                    "score": pl.Float64,
                    "should_blur": pl.Boolean,
                }
            )

        outputs: list[dict] = []
        ema_alpha = np.clip(self.params.ema_alpha, 0.0, 1.0)
        dilation_pct = max(0.0, self.params.bbox_dilate_pct)

        for frame_key, frame_df in detections.group_by("frame", maintain_order=True):
            frame_idx = frame_key[0] if isinstance(frame_key, tuple) else int(frame_key)
            rows = [row for row in frame_df.iter_rows(named=True) if row.get("is_confident", True)]
            # Drop EMA state for detections not present this frame
            self._prune_state(len(rows))

            for det_index, row in enumerate(rows):
                tlwh = np.array(
                    [
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]) - float(row["x1"]),
                        float(row["y2"]) - float(row["y1"]),
                    ],
                    dtype=float,
                )
                # Apply EMA smoothing per detection index
                prev = self._ema_state.get(det_index)
                smoothed = tlwh if prev is None else ema_alpha * tlwh + (1.0 - ema_alpha) * prev
                self._ema_state[det_index] = smoothed

                dilated = self._dilate_box(smoothed, dilation_pct)

                track_id = self._track_ids.get(det_index)
                if track_id is None:
                    track_id = self._next_track_id
                    self._next_track_id += 1
                    self._track_ids[det_index] = track_id

                observation = TrackObservation(
                    frame=frame_idx,
                    track_id=track_id,
                    tlwh=dilated,
                    state=TrackState.CONFIRMED,
                    age=1,
                    last_seen=frame_idx,
                    score=float(row.get("confidence", 1.0)),
                    should_blur=True,
                )
                outputs.append(observation.as_dict())
                self._timeline.append(observation)

        return pl.DataFrame(outputs)

    def get_tracker_info(self) -> dict[str, object]:
        return {
            "tracker_type": "DummyTracker",
            "params": self.params.model_dump(),
        }

    def _prune_state(self, count: int) -> None:
        excess_keys = [idx for idx in self._ema_state if idx >= count]
        for idx in excess_keys:
            self._ema_state.pop(idx, None)
            self._track_ids.pop(idx, None)

    @staticmethod
    def _dilate_box(tlwh: np.ndarray, pct: float) -> np.ndarray:
        x, y, w, h = tlwh
        dw = w * pct
        dh = h * pct
        return np.array([x - dw / 2.0, y - dh / 2.0, w + dw, h + dh], dtype=float)

    # ------------------------------------------------------------------
    # Accessors (match BaseTracker subset)
    # ------------------------------------------------------------------
    @property
    def track_history(self) -> list[TrackObservation]:
        """Return a shallow copy of the emitted observations for debug overlays."""
        return list(self._timeline)


__all__ = ["DummyTracker"]
