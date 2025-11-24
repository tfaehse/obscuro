from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from anonymizer.config import TrackerParams

from .base import BaseTracker
from .common import Detection, TrackObservation, TrackState, batched_center_distance
from .embeddings import get_embedding_model
from .utils import clamp_bbox, cosine_similarities, crop_patch, update_weighted_embedding


class FusedTracker(BaseTracker):
    """
    ByteTrack-style tracker augmented with appearance gating.
    Uses high/low confidence pools, combines distance, shape, and embedding similarity.
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
        self._embedding_model = get_embedding_model()
        self._high_score_threshold = confidence_threshold
        self._low_score_threshold = low_score_threshold

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

    def _process_frame(
        self,
        frame_idx: int,
        detections: Sequence[Detection],
        low_conf_detections: Sequence[Detection] | None = None,
    ) -> list[TrackObservation]:
        self._predict_tracks()

        frame = self._prepare_frame(self._get_frame(frame_idx))
        detections = [*detections, *(low_conf_detections or [])]
        det_embeddings = self._compute_embeddings(frame, detections)

        # Split pools like ByteTrack
        high_indices: list[int] = []
        low_indices: list[int] = []
        low_thresh = max(0.0, self._low_score_threshold) if self.params.use_low_score_pool else 1.1
        for idx, det in enumerate(detections):
            if det.score >= self._high_score_threshold:
                high_indices.append(idx)
            elif det.score >= low_thresh:
                low_indices.append(idx)

        matches: list[tuple[int, int]] = []
        matches_hi, unmatched_tracks, unmatched_high = self._match_with_cost(
            range(len(self._tracks)),
            high_indices,
            detections,
            det_embeddings,
            distance_gate=self.params.distance_gate_hi or self.params.distance_gate,
        )
        matches.extend(matches_hi)

        matched_high = {idx for _, idx in matches_hi}

        if low_indices and unmatched_tracks:
            matches_low, unmatched_tracks, _ = self._match_with_cost(
                unmatched_tracks,
                low_indices,
                detections,
                det_embeddings,
                distance_gate=self.params.distance_gate_lo or self.params.distance_gate,
            )
            matches.extend(matches_low)

        observations: list[TrackObservation] = []

        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            det = detections[det_idx]
            self._update_track(track, det, frame_idx)
            if det_embeddings and det_idx < len(det_embeddings):
                emb = det_embeddings[det_idx]
                if emb is not None:
                    track.vt_embeddings = [*track.vt_embeddings, emb][-3:]
                    track.vt_embedding_rep = update_weighted_embedding(
                        track.vt_embedding_rep, emb, det.score
                    )
            obs = self._emit_observation(track, frame_idx, matched=True)
            if obs:
                observations.append(obs)

        for idx in unmatched_tracks:
            track = self._tracks[idx]
            self._mark_missed(track, frame_idx)
            obs = self._emit_observation(track, frame_idx, matched=False)
            if obs:
                observations.append(obs)

        unmatched_high = [i for i in unmatched_high if i not in matched_high]
        for det_idx in unmatched_high:
            obs = self._start_new_track(detections[det_idx], frame_idx)
            if obs:
                if det_embeddings and det_idx < len(det_embeddings):
                    emb = det_embeddings[det_idx]
                    if emb is not None and self._tracks:
                        self._tracks[-1].vt_embeddings = [emb]
                        self._tracks[-1].vt_embedding_rep = emb / (np.linalg.norm(emb) + 1e-8)
                observations.append(obs)

        survivors = []
        for track in self._tracks:
            if track.state == TrackState.DEAD:
                self._retired_tracks.append(track)
            else:
                survivors.append(track)
        self._tracks = survivors
        return observations

    def _compute_embeddings(
        self, frame: np.ndarray | None, detections: Sequence[Detection]
    ) -> list[np.ndarray | None]:
        if self._embedding_model is None or frame is None:
            return [None] * len(detections)
        embs: list[np.ndarray | None] = []
        for det in detections:
            tlwh = det.tlwh.astype(int)
            bbox = (int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3]))
            clamped = clamp_bbox(bbox, frame.shape)
            if clamped is None:
                embs.append(None)
                continue
            patch = crop_patch(frame, clamped)
            if patch is None or patch.size == 0:
                embs.append(None)
                continue
            try:
                embs.append(self._embedding_model.embed(patch))
            except Exception:
                embs.append(None)
        return embs

    def _match_with_cost(
        self,
        track_indices: Iterable[int],
        det_indices: Iterable[int],
        detections: Sequence[Detection],
        det_embeddings: Sequence[np.ndarray | None],
        *,
        distance_gate: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        track_indices = list(track_indices)
        det_indices = list(det_indices)
        if not track_indices or not det_indices:
            return [], track_indices, det_indices

        track_boxes = [self._tracks[idx].current_tlwh() for idx in track_indices]
        det_boxes = [detections[idx].tlwh for idx in det_indices]
        track_sizes = [self._tracks[idx].frame_size for idx in track_indices]
        det_sizes = [detections[idx].frame_size for idx in det_indices]

        distance_matrix = batched_center_distance(track_boxes, track_sizes, det_boxes, det_sizes)
        cost_matrix = np.full_like(distance_matrix, distance_gate + 10.0, dtype=float)

        for r, track_idx in enumerate(track_indices):
            t_box = track_boxes[r]
            t_asp = max(1e-6, t_box[2] / max(t_box[3], 1e-6))
            t_embs = getattr(self._tracks[track_idx], "vt_embeddings", [])
            t_rep = getattr(self._tracks[track_idx], "vt_embedding_rep", None)
            for c, det_idx in enumerate(det_indices):
                dist = distance_matrix[r, c]
                if not np.isfinite(dist) or dist > distance_gate:
                    continue
                d_box = det_boxes[c]
                d_asp = max(1e-6, d_box[2] / max(d_box[3], 1e-6))
                shape_penalty = min(1.0, abs(t_asp - d_asp))
                cost = dist + 0.1 * shape_penalty
                det_emb = det_embeddings[det_idx]
                if det_emb is not None:
                    det_emb = det_emb.astype(np.float32)
                    if t_rep is not None:
                        rep_norm = t_rep / (np.linalg.norm(t_rep) + 1e-8)
                        det_norm = det_emb / (np.linalg.norm(det_emb) + 1e-8)
                        sim_rep = float(np.dot(rep_norm, det_norm))
                        if sim_rep < self.params.embedding_similarity_gate:
                            continue
                        cost += (1.0 - sim_rep) * 0.3
                    if t_embs:
                        sims = cosine_similarities(t_embs, det_emb)
                        max_sim = float(np.max(sims))
                        if max_sim < self.params.embedding_similarity_gate:
                            continue
                        cost += (1.0 - max_sim) * 0.2
                cost_matrix[r, c] = cost

        row_ind = col_ind = np.array([], dtype=int)
        if cost_matrix.size:
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(track_indices)
        unmatched_dets = set(det_indices)

        for r, c in zip(row_ind, col_ind, strict=False):
            cost = cost_matrix[r, c]
            if cost > distance_gate + 1.0:
                continue
            t_idx = track_indices[r]
            d_idx = det_indices[c]
            matches.append((t_idx, d_idx))
            unmatched_tracks.discard(t_idx)
            unmatched_dets.discard(d_idx)

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


__all__ = ["FusedTracker"]
