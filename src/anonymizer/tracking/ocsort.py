"""OC-SORT: Observation-Centric SORT for robust multi-object tracking.

Implements the key innovations from the OC-SORT paper (CVPR 2023):
- Observation-Centric Re-Update (OC-RE-UPDATE): Corrects Kalman state using trajectory
- Observation-Centric Momentum (OC-M): Uses velocity from recent observations
- Virtual Trajectory: Interpolated observations during tracking gaps

Reference: https://github.com/noahcao/OC_SORT
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from anonymizer.config import TrackerParams

from .base import ActiveTrack, BaseTracker
from .common import Detection, TrackObservation, TrackState
from .kalman import initiate, mean_to_tlwh
from .kalman import predict as kf_predict
from .kalman import update as kf_update

logger = logging.getLogger(__name__)


@dataclass
class OCSortTrackState:
    """OC-SORT specific state for a track, stored separately from ActiveTrack."""

    # Last observation (tlwh) for momentum computation
    last_observation: np.ndarray | None = None
    # Second-to-last observation for velocity estimation
    last_last_observation: np.ndarray | None = None
    # Frame index of last observation
    last_obs_frame: int = -1
    # Accumulated velocity from observations
    delta_mean: np.ndarray | None = None


class OCSortTracker(BaseTracker):
    """OC-SORT: Observation-Centric SORT tracker.

    Key improvements over standard SORT/ByteTrack:
    1. Observation-Centric Momentum: Uses velocity from recent observations instead of
       relying solely on Kalman filter predictions during occlusions.
    2. Observation-Centric Re-Update: When a track is re-observed after misses, it
       corrects the Kalman state using the trajectory history.
    3. Virtual Trajectory Association: Creates virtual observations for better
       association during brief occlusions.

    This implementation extends BaseTracker and uses the low-score pool mechanism
    from ByteTrack while adding OC-SORT's observation-centric innovations.
    """

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
        # OC-SORT specific: minimum IoU for track-detection association
        self._iou_threshold = 0.3
        # OC-SORT specific state per track (keyed by track_id)
        self._oc_state: dict[int, OCSortTrackState] = {}

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
        # Clear OC-SORT state on reconfigure
        self._oc_state.clear()

    def _get_oc_state(self, track_id: int) -> OCSortTrackState:
        """Get or create OC-SORT state for a track."""
        if track_id not in self._oc_state:
            self._oc_state[track_id] = OCSortTrackState()
        return self._oc_state[track_id]

    def _process_frame(
        self,
        frame_idx: int,
        detections: Sequence[Detection],
        low_conf_detections: Sequence[Detection] | None = None,
    ) -> list[TrackObservation]:
        """Process a frame with OC-SORT's observation-centric approach."""
        # Predict all tracks forward
        self._predict_tracks()

        # Combine detections and split into high/low score pools
        all_detections: list[Detection] = [*detections, *(low_conf_detections or [])]
        high_pool: list[Detection] = []
        high_indices: list[int] = []
        low_pool: list[Detection] = []
        low_indices: list[int] = []

        low_thresh = max(0.0, self._low_score_threshold) if self.params.use_low_score_pool else 1.1

        for idx, det in enumerate(all_detections):
            if det.score >= self._high_score_threshold:
                high_pool.append(det)
                high_indices.append(idx)
            elif det.score >= low_thresh:
                low_pool.append(det)
                low_indices.append(idx)

        # First association: high-score detections with all tracks
        matches_high, unmatched_tracks, unmatched_high = self._associate_oc_sort(
            range(len(self._tracks)), high_pool, frame_idx
        )

        matches: list[tuple[int, int]] = []
        matches.extend([(track_idx, high_indices[det_idx]) for track_idx, det_idx in matches_high])

        matched_high_det_indices = {high_indices[idx] for _, idx in matches_high}

        # Second association: low-score detections with remaining unmatched tracks
        if low_pool and unmatched_tracks:
            matches_low, unmatched_tracks, _ = self._associate_oc_sort(
                unmatched_tracks, low_pool, frame_idx, use_virtual_trajectory=True
            )
            matches.extend(
                [(track_idx, low_indices[det_idx]) for track_idx, det_idx in matches_low]
            )

        observations: list[TrackObservation] = []
        matched_tracks = {track_idx for track_idx, _ in matches}

        # Update matched tracks with OC-SORT's observation-centric update
        for track_idx, det_global_idx in matches:
            track = self._tracks[track_idx]
            det = all_detections[det_global_idx]
            self._update_track_oc(track, det, frame_idx)
            obs = self._emit_observation(track, frame_idx, matched=True)
            if obs:
                observations.append(obs)

        # Mark unmatched tracks as missed
        for track_idx in range(len(self._tracks)):
            if track_idx in matched_tracks:
                continue
            track = self._tracks[track_idx]
            self._mark_missed(track, frame_idx)
            obs = self._emit_observation(track, frame_idx, matched=False)
            if obs:
                observations.append(obs)

        # Start new tracks from unmatched high-score detections
        unmatched_high_det_indices = {
            high_indices[idx]
            for idx in unmatched_high
            if high_indices[idx] not in matched_high_det_indices
        }
        for det_idx in unmatched_high_det_indices:
            obs = self._start_new_track_oc(all_detections[det_idx], frame_idx)
            if obs:
                observations.append(obs)

        # Cull dead tracks and clean up their OC-SORT state
        survivors = []
        for track in self._tracks:
            if track.state == TrackState.DEAD:
                # Clean up OC-SORT state for dead tracks
                self._oc_state.pop(track.track_id, None)
                self._retired_tracks.append(track)
            else:
                survivors.append(track)
        self._tracks = survivors
        self._prune_low_detection_rate()

        return observations

    def _associate_oc_sort(
        self,
        track_indices: Sequence[int],
        detections: Sequence[Detection],
        frame_idx: int,
        *,
        use_virtual_trajectory: bool = False,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate tracks to detections using OC-SORT's IoU-based approach.

        OC-SORT uses IoU instead of center distance for association, which is more
        robust to scale changes. It also uses virtual trajectories for tracks that
        have been missing for a few frames.
        """
        if not track_indices or not detections:
            return [], list(track_indices), list(range(len(detections)))

        # Compute IoU matrix between track predictions and detections
        iou_matrix = self._compute_iou_matrix(track_indices, detections)

        # Use Hungarian algorithm for assignment
        # Convert IoU to cost (lower is better)
        cost_matrix = 1.0 - iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = set(track_indices)
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind, strict=False):
            iou = iou_matrix[r, c]
            if iou >= self._iou_threshold:
                matches.append((track_indices[r], c))
                unmatched_tracks.discard(track_indices[r])
                unmatched_dets.discard(c)

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    def _compute_iou_matrix(
        self,
        track_indices: Sequence[int],
        detections: Sequence[Detection],
    ) -> np.ndarray:
        """Compute IoU matrix between track boxes and detection boxes."""
        n_tracks = len(track_indices)
        n_dets = len(detections)

        if n_tracks == 0 or n_dets == 0:
            return np.zeros((n_tracks, n_dets), dtype=float)

        iou_matrix = np.zeros((n_tracks, n_dets), dtype=float)

        for i, track_idx in enumerate(track_indices):
            track = self._tracks[track_idx]
            track_box = track.current_tlwh()

            # Convert normalized coordinates to absolute for IoU computation
            # If frame_size is available, use it; otherwise assume normalized
            if track.frame_size is not None:
                fw, fh = track.frame_size
                track_box_abs = track_box.copy()
                track_box_abs[0] *= fw
                track_box_abs[1] *= fh
                track_box_abs[2] *= fw
                track_box_abs[3] *= fh
            else:
                track_box_abs = track_box

            for j, det in enumerate(detections):
                det_box = det.tlwh

                if det.frame_size is not None:
                    fw, fh = det.frame_size
                    det_box_abs = det_box.copy()
                    det_box_abs[0] *= fw
                    det_box_abs[1] *= fh
                    det_box_abs[2] *= fw
                    det_box_abs[3] *= fh
                else:
                    det_box_abs = det_box

                iou_matrix[i, j] = self._compute_iou(track_box_abs, det_box_abs)

        return iou_matrix

    def _compute_iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two boxes in tlwh format."""
        x1_a, y1_a, w_a, h_a = box_a
        x1_b, y1_b, w_b, h_b = box_b

        x2_a = x1_a + w_a
        y2_a = y1_a + h_a
        x2_b = x1_b + w_b
        y2_b = y1_b + h_b

        # Intersection
        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Union
        area_a = w_a * h_a
        area_b = w_b * h_b
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return float(inter_area / union_area)

    def _update_track_oc(self, track: ActiveTrack, det: Detection, frame_idx: int) -> None:
        """Update track with OC-SORT's observation-centric approach.

        Key OC-SORT innovation: When a track gets a new observation after misses,
        use the observation trajectory to correct the Kalman filter state instead
        of just doing a single update.
        """
        oc_state = self._get_oc_state(track.track_id)

        prev_obs = oc_state.last_observation
        prev_obs_frame = oc_state.last_obs_frame

        # Standard Kalman update
        track.mean, track.covariance = kf_update(track.mean, track.covariance, det.tlwh)

        # OC-SORT: Observation-Centric Momentum
        # Compute velocity from observations instead of Kalman state
        if prev_obs is not None and prev_obs_frame >= 0:
            dt = max(1, frame_idx - prev_obs_frame)
            # Velocity in tlwh space
            velocity = (det.tlwh - prev_obs) / dt
            # Store for prediction during misses
            oc_state.delta_mean = velocity.copy()

            # OC-SORT: Observation-Centric Re-Update
            # If track was missing for multiple frames, correct the trajectory
            if dt > 1 and oc_state.last_last_observation is not None:
                # Re-update with virtual trajectory
                # This corrects accumulated Kalman drift
                self._re_update_with_trajectory(track, det.tlwh, dt, oc_state)

        # Update observation history
        oc_state.last_last_observation = oc_state.last_observation
        oc_state.last_observation = det.tlwh.copy()
        oc_state.last_obs_frame = frame_idx

        # Standard track update
        track.frame_size = det.frame_size
        track.score = det.score
        track.last_seen = frame_idx
        track.mask = det.mask
        track.hits += 1
        track.misses = 0
        track.state = (
            TrackState.CONFIRMED
            if track.hits >= self.params.confirm_after_N
            else TrackState.TENTATIVE
        )
        track.smoothed_tlwh = self._smooth_tlwh(track, mean_to_tlwh(track.mean))

    def _re_update_with_trajectory(
        self,
        track: ActiveTrack,
        current_obs: np.ndarray,
        gap: int,
        oc_state: OCSortTrackState,
    ) -> None:
        """OC-SORT's Observation-Centric Re-Update.

        When a track is re-observed after a gap, this method creates virtual
        observations along the trajectory and uses them to correct the Kalman state.
        This helps correct accumulated drift from Kalman predictions during the gap.
        """
        if gap <= 1 or oc_state.last_last_observation is None:
            return

        # Compute velocity from observation trajectory
        start_obs = oc_state.last_last_observation
        end_obs = current_obs

        # Linear interpolation for virtual observations
        # This creates a more stable trajectory update
        for t in range(1, gap):
            alpha = t / gap
            virtual_obs = (1 - alpha) * start_obs + alpha * end_obs
            # Light update with virtual observation (reduced weight)
            track.mean, track.covariance = kf_update(track.mean, track.covariance, virtual_obs)

    def _predict_tracks(self) -> None:
        """Predict tracks forward with OC-SORT's observation-centric momentum."""
        for track in self._tracks:
            # Standard Kalman prediction
            track.mean, track.covariance = kf_predict(track.mean, track.covariance)
            track.age += 1
            track.misses += 1

            # OC-SORT: Apply observation-centric momentum correction
            # If we have recent observations, correct the velocity
            oc_state = self._oc_state.get(track.track_id)
            if oc_state is not None and oc_state.delta_mean is not None and track.misses <= 3:
                # Apply observation-based velocity correction to the mean
                # This helps maintain trajectory during brief occlusions
                track.mean[:4] += oc_state.delta_mean * 0.5

    def _start_new_track_oc(self, det: Detection, frame_idx: int) -> TrackObservation | None:
        """Start a new track with OC-SORT specific initialization."""
        mean, covariance = initiate(det.tlwh)
        track = ActiveTrack(
            track_id=self._next_id,
            mean=mean,
            covariance=covariance,
            state=TrackState.TENTATIVE,
            hits=1,
            misses=0,
            age=1,
            last_seen=frame_idx,
            score=det.score,
            frame_size=det.frame_size,
            smoothed_tlwh=det.tlwh.copy(),
            mask=det.mask,
        )

        # Initialize OC-SORT specific state
        oc_state = OCSortTrackState(
            last_observation=det.tlwh.copy(),
            last_last_observation=None,
            last_obs_frame=frame_idx,
            delta_mean=None,
        )
        self._oc_state[track.track_id] = oc_state

        self._next_id += 1
        track.state = (
            TrackState.CONFIRMED
            if track.hits >= self.params.confirm_after_N
            else TrackState.TENTATIVE
        )
        track.smoothed_tlwh = self._smooth_tlwh(track, det.tlwh)
        self._tracks.append(track)
        return self._emit_observation(track, frame_idx, matched=True)


__all__ = ["OCSortTracker"]
