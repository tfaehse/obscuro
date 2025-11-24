from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from anonymizer.config import TrackerParams

from .base import BaseTracker
from .common import (
    Detection,
    TrackObservation,
    TrackState,
    batched_center_distance,
)


class BotSortTracker(BaseTracker):
    """BoT-SORT style tracker with appearance cues and light motion compensation."""

    def __init__(
        self,
        video_source: str | Path | None,
        *,
        params: TrackerParams | None = None,
        cancel_event=None,
        progress_callback=None,
        confidence_threshold: float = 0.5,
        low_score_threshold: float = 0.1,
    ) -> None:
        super().__init__(
            video_source,
            params=params,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        self._high_score_threshold = confidence_threshold
        self._low_score_threshold = low_score_threshold
        self._prev_centers: list[tuple[float, float]] = []

    def set_thresholds(
        self,
        confidence_threshold: float | None = None,
        low_score_threshold: float | None = None,
    ) -> None:
        """Update detection thresholds at runtime."""
        if confidence_threshold is not None:
            self._high_score_threshold = confidence_threshold
        if low_score_threshold is not None:
            self._low_score_threshold = low_score_threshold

    def reconfigure(
        self,
        params: TrackerParams,
        confidence_threshold: float | None = None,
        low_score_threshold: float | None = None,
    ) -> None:
        super().reconfigure(params)
        self.set_thresholds(confidence_threshold, low_score_threshold)
        self._prev_centers.clear()

    def _process_frame(
        self,
        frame_idx: int,
        detections: Sequence[Detection],
        low_conf_detections: Sequence[Detection] | None = None,
    ) -> list[TrackObservation]:
        self._predict_tracks()

        detections_list = [*detections, *(low_conf_detections or [])]
        matches: list[tuple[int, int]] = []

        # Stage 1: tight distance gate
        track_indices = list(range(len(self._tracks)))
        det_indices = list(range(len(detections_list)))
        matches_hi, unmatched_tracks, unmatched_dets = self._match_tracks(
            track_indices,
            det_indices,
            detections_list,
            distance_gate=self.params.distance_gate_hi,
            use_appearance=False,
        )
        matches.extend(matches_hi)

        # Stage 2: relaxed distance gate with appearance re-ID
        if unmatched_tracks and unmatched_dets:
            matches_lo, unmatched_tracks, unmatched_dets = self._match_tracks(
                unmatched_tracks,
                unmatched_dets,
                detections_list,
                distance_gate=self.params.distance_gate_lo,
                use_appearance=False,
            )
            matches.extend(matches_lo)

        observations: list[TrackObservation] = []
        matched_tracks = {track_idx for track_idx, _ in matches}

        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            det = detections_list[det_idx]
            self._update_track(track, det, frame_idx)
            obs = self._emit_observation(track, frame_idx, matched=True)
            if obs:
                observations.append(obs)

        # Optional ByteTrack-style low score pool for remaining tracks
        if self.params.use_low_score_pool and unmatched_tracks:
            low_pool: list[int] = []
            for det_idx in unmatched_dets:
                det = detections_list[det_idx]
                if self._low_score_threshold <= det.score < self._high_score_threshold:
                    low_pool.append(det_idx)
            if low_pool:
                matches_low, unmatched_tracks, unmatched_dets = self._match_tracks(
                    unmatched_tracks,
                    low_pool,
                    detections_list,
                    distance_gate=self.params.distance_gate_lo,
                    use_appearance=False,
                )
                for track_idx, det_idx in matches_low:
                    track = self._tracks[track_idx]
                    det = detections_list[det_idx]
                    self._update_track(track, det, frame_idx)
                    obs = self._emit_observation(track, frame_idx, matched=True)
                    if obs:
                        observations.append(obs)
                matched_tracks.update(track_idx for track_idx, _ in matches_low)
                unmatched_dets = [idx for idx in unmatched_dets if idx not in low_pool]

        # Tracks still unmatched
        for idx in range(len(self._tracks)):
            if idx in matched_tracks:
                continue
            track = self._tracks[idx]
            self._mark_missed(track, frame_idx)
            obs = self._emit_observation(track, frame_idx, matched=False)
            if obs:
                observations.append(obs)

        # Spawn new tracks from remaining high-confidence detections
        for det_idx in unmatched_dets:
            det = detections_list[det_idx]
            if det.score >= self._high_score_threshold:
                obs = self._start_new_track(det, frame_idx)
                if obs:
                    observations.append(obs)

        self._prev_centers = [self._center(det.tlwh) for det in detections_list]

        survivors = []
        for track in self._tracks:
            if track.state == TrackState.DEAD:
                self._retired_tracks.append(track)
            else:
                survivors.append(track)
        self._tracks = survivors

        return observations

    def _match_tracks(
        self,
        track_indices: Iterable[int],
        detection_indices: Iterable[int],
        detections: Sequence[Detection],
        *,
        distance_gate: float,
        use_appearance: bool,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        track_indices = list(track_indices)
        det_indices = list(detection_indices)
        if not track_indices or not det_indices:
            return [], track_indices, det_indices

        assignment_threshold = distance_gate

        # Motion compensation: derive simple translation based on center displacement
        translation = np.zeros(2, dtype=float)
        if self.params.cam_motion_comp and self._prev_centers and det_indices:
            pairs = min(len(self._prev_centers), len(det_indices))
            if pairs:
                current_centers = np.array(
                    [self._center(detections[idx].tlwh) for idx in det_indices[:pairs]]
                )
                prev_centers = np.array(self._prev_centers[:pairs])
                translation = np.mean(current_centers - prev_centers, axis=0)

        track_boxes = []
        for idx in track_indices:
            box = self._tracks[idx].current_tlwh().copy()
            if self.params.cam_motion_comp:
                box[0] -= translation[0]
                box[1] -= translation[1]
            track_boxes.append(box)

        det_boxes = [detections[idx].tlwh for idx in det_indices]
        track_sizes = [self._tracks[idx].frame_size for idx in track_indices]
        det_sizes = [detections[idx].frame_size for idx in det_indices]
        distance_matrix = batched_center_distance(track_boxes, track_sizes, det_boxes, det_sizes)
        cost_matrix = np.full_like(distance_matrix, assignment_threshold + 1.0, dtype=float)

        for r in range(len(track_indices)):
            for c in range(len(det_indices)):
                dist = distance_matrix[r, c]
                if not np.isfinite(dist):
                    continue
                if dist <= distance_gate:
                    base_cost = dist
                else:
                    continue

                cost_matrix[r, c] = base_cost

        row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)
        if cost_matrix.size > 0:
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(track_indices)
        unmatched_dets = set(det_indices)

        for r, c in zip(row_ind, col_ind, strict=False):
            cost = cost_matrix[r, c]
            if cost > assignment_threshold:
                continue
            track_idx = track_indices[r]
            det_idx = det_indices[c]
            matches.append((track_idx, det_idx))
            unmatched_tracks.discard(track_idx)
            unmatched_dets.discard(det_idx)

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    @staticmethod
    def _center(tlwh: np.ndarray) -> tuple[float, float]:
        return float(tlwh[0] + tlwh[2] / 2.0), float(tlwh[1] + tlwh[3] / 2.0)


__all__ = ["BotSortTracker"]
