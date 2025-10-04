from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from anonymizer.config import TrackerParams

from .base import BaseTracker
from .common import Detection, TrackObservation, TrackState


class ByteTrackTracker(BaseTracker):
    """ByteTrack keeps tracks alive with a secondary low-score detection pool."""

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
        self._high_score_threshold = self.params.high_thresh
        self._low_score_threshold = self.params.low_thresh

    def reconfigure(self, params: TrackerParams) -> None:
        super().reconfigure(params)
        self._high_score_threshold = self.params.high_thresh
        self._low_score_threshold = self.params.low_thresh

    def _process_frame(
        self,
        frame_idx: int,
        detections: Sequence[Detection],
        low_conf_detections: Sequence[Detection] | None = None,
    ) -> list[TrackObservation]:
        self._predict_tracks()

        high_pool: list[Detection] = []
        high_indices: list[int] = []
        low_pool: list[Detection] = []
        low_indices: list[int] = []

        low_thresh = (
            max(0.0, self._low_score_threshold) if self.params.use_low_score_pool else 1.1
        )  # disable low pool

        for idx, det in enumerate(detections):
            if det.score >= self._high_score_threshold:
                high_pool.append(det)
                high_indices.append(idx)
            elif det.score >= low_thresh:
                low_pool.append(det)
                low_indices.append(idx)

        matches: list[tuple[int, int]] = []
        unmatched_tracks: list[int]
        unmatched_high: list[int]

        matches_high, unmatched_tracks, unmatched_high = self._associate_subset(
            range(len(self._tracks)), high_pool
        )
        matches.extend([(track_idx, high_indices[det_idx]) for track_idx, det_idx in matches_high])

        matched_high_det_indices = {high_indices[idx] for _, idx in matches_high}

        # Try to keep unmatched tracks alive with low-score detections
        if low_pool and unmatched_tracks:
            matches_low, unmatched_tracks, _ = self._associate_subset(unmatched_tracks, low_pool)
            matches.extend(
                [(track_idx, low_indices[det_idx]) for track_idx, det_idx in matches_low]
            )

        observations: list[TrackObservation] = []
        matched_tracks = {track_idx for track_idx, _ in matches}

        for track_idx, det_global_idx in matches:
            track = self._tracks[track_idx]
            det = detections[det_global_idx]
            self._update_track(track, det, frame_idx)
            obs = self._emit_observation(track, frame_idx, matched=True)
            if obs:
                observations.append(obs)

        # Tracks that remain unmatched after both pools
        for track_idx in range(len(self._tracks)):
            if track_idx in matched_tracks:
                continue
            track = self._tracks[track_idx]
            self._mark_missed(track, frame_idx)
            obs = self._emit_observation(track, frame_idx, matched=False)
            if obs:
                observations.append(obs)

        # Spawn new tracks only from remaining high-score detections
        unmatched_high_det_indices = {
            high_indices[idx]
            for idx in unmatched_high
            if high_indices[idx] not in matched_high_det_indices
        }
        for det_idx in unmatched_high_det_indices:
            obs = self._start_new_track(detections[det_idx], frame_idx)
            if obs:
                observations.append(obs)

        survivors = []
        for track in self._tracks:
            if track.state == TrackState.DEAD:
                self._retired_tracks.append(track)
            else:
                survivors.append(track)
        self._tracks = survivors

        return observations


__all__ = ["ByteTrackTracker"]
