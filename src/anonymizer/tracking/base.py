from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from anonymizer.config import TrackerParams
from anonymizer.io.video import iter_frames
from anonymizer.utils.progress import ProgressRateEstimator, format_progress_message

from .common import Detection, TrackObservation, TrackState, batched_center_distance
from .kalman import initiate, mean_to_tlwh
from .kalman import predict as kf_predict
from .kalman import update as kf_update
from .utils import prepare_frame_rgb

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ActiveTrack:
    """Internal track state."""

    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    state: TrackState
    hits: int = 0
    misses: int = 0
    age: int = 0
    last_seen: int = -1
    score: float = 0.0
    smoothed_tlwh: np.ndarray | None = None
    frame_size: tuple[int, int] | None = None
    history: list[TrackObservation] = field(default_factory=list)
    visual_tracker: object | None = None
    debug_color: tuple[int, int, int] | None = None
    vt_embeddings: list[np.ndarray] = field(default_factory=list, repr=False)
    vt_embedding_rep: np.ndarray | None = field(default=None, repr=False)
    mask: dict[str, object] | None = None

    def current_tlwh(self) -> np.ndarray:
        if self.smoothed_tlwh is not None:
            return self.smoothed_tlwh.copy()
        return mean_to_tlwh(self.mean)


class BaseTracker:
    """Unified online tracking pipeline implementing predict → associate → update."""

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
        self.video_source = Path(video_source) if video_source else None
        self.params = params or TrackerParams()
        self.cancel_event = cancel_event
        self.progress_callback = progress_callback
        self._frame_iter: Iterator[tuple[int, np.ndarray]] | None = None
        self._last_frame: np.ndarray | None = None
        self._last_frame_idx: int | None = None

        self._tracks: list[ActiveTrack] = []
        self._retired_tracks: list[ActiveTrack] = []
        self._timeline: list[TrackObservation] = []
        self._next_id = 1
        self._frames_processed = 0
        self._high_score_threshold = confidence_threshold
        self._low_score_threshold = low_score_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_video_source(self, source: Path | str | None) -> None:
        self.video_source = Path(source) if source else None
        self._reset_frame_iterator()

    def reconfigure(
        self,
        params: TrackerParams,
        confidence_threshold: float | None = None,
        low_score_threshold: float | None = None,
    ) -> None:
        """Reconfigure tracker parameters. Subclasses may use threshold params."""
        logger.info("Reconfiguring tracker parameters: %s", params.model_dump())
        self.params = params
        self._reset_frame_iterator()
        self.set_thresholds(confidence_threshold, low_score_threshold)

    def set_thresholds(
        self,
        confidence_threshold: float | None = None,
        low_score_threshold: float | None = None,
    ) -> None:
        """Update detection thresholds; base stores for subclasses that consult them."""
        if confidence_threshold is not None:
            self._high_score_threshold = confidence_threshold
        if low_score_threshold is not None:
            self._low_score_threshold = low_score_threshold

    def track(self, detections: pl.DataFrame) -> pl.DataFrame:
        if detections.is_empty():
            self._tracks.clear()
            self._retired_tracks.clear()
            self._timeline.clear()
            return pl.DataFrame(
                schema={
                    "frame": pl.Int64,
                    "track_id": pl.Int64,
                    "x1": pl.Float64,
                    "y1": pl.Float64,
                    "x2": pl.Float64,
                    "y2": pl.Float64,
                    "state": pl.Utf8,
                    "age": pl.Int64,
                    "last_seen": pl.Int64,
                    "score": pl.Float64,
                    "mask": pl.Int64,
                }
            )

        self._timeline.clear()

        detections = detections.sort("frame")
        outputs: list[dict] = []
        total_frames = (
            detections.get_column("frame").n_unique()
            if "frame" in detections.columns
            else detections.height
        )
        processed_frames = 0
        rate_tracker = ProgressRateEstimator()

        for frame_key, frame_df in detections.group_by("frame", maintain_order=True):
            frame_start = time.perf_counter()
            frame_idx = frame_key[0] if isinstance(frame_key, tuple) else int(frame_key)
            frame_detections, low_conf_detections = self._polars_to_detections(frame_idx, frame_df)
            frame_outputs = self._process_frame(frame_idx, frame_detections, low_conf_detections)
            for obs in frame_outputs:
                self._timeline.append(obs)
                if obs.should_blur:
                    outputs.append(obs.as_dict())

            processed_frames += 1
            self._frames_processed += 1
            duration = max(1e-6, time.perf_counter() - frame_start)
            fps = rate_tracker.record(1, duration)
            if self.progress_callback:
                remaining = max(total_frames - processed_frames, 0) if total_frames > 0 else None
                if total_frames > 0:
                    percentage = int(min(100, (processed_frames / total_frames) * 100))
                    prefix = f"Processed frame {processed_frames}/{total_frames}"
                else:
                    percentage = min(99, processed_frames)
                    prefix = f"Processed frame {processed_frames}"
                message = format_progress_message(prefix, fps, remaining)
                self.progress_callback(percentage, "Tracking", message)

        df = pl.DataFrame(outputs) if outputs else pl.DataFrame({})
        if "mask" in df.columns:
            df = df.with_columns(pl.col("mask").cast(pl.Int64))
        return df

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------
    def _reset_frame_iterator(self) -> None:
        if self._frame_iter is not None:
            closer = getattr(self._frame_iter, "close", None)
            if callable(closer):
                with contextlib.suppress(Exception):
                    closer()
        self._frame_iter = None
        self._last_frame = None
        self._last_frame_idx = None

    def _ensure_frame_iterator(self) -> None:
        if self._frame_iter is None and self.video_source is not None:
            with contextlib.suppress(Exception):
                self._frame_iter = iter_frames(self.video_source, prefetch=1)

    def _get_frame(self, frame_idx: int) -> np.ndarray | None:
        if self.video_source is None:
            return None
        self._ensure_frame_iterator()
        if self._frame_iter is None:
            return None
        if self._last_frame_idx is not None and frame_idx <= self._last_frame_idx:
            return self._last_frame
        try:
            while self._last_frame_idx is None or self._last_frame_idx < frame_idx:
                next_idx, frame = next(self._frame_iter)
                self._last_frame = frame
                self._last_frame_idx = next_idx
        except StopIteration:
            self._reset_frame_iterator()
            return None
        return self._last_frame

    def _prepare_frame(self, frame: np.ndarray | None) -> np.ndarray | None:
        return prepare_frame_rgb(frame)

    def _process_frame(
        self,
        frame_idx: int,
        detections: Sequence[Detection],
        low_conf_detections: Sequence[Detection] | None = None,
    ) -> list[TrackObservation]:
        self._predict_tracks()

        matches, unmatched_tracks, unmatched_detections = self._associate(detections)

        observations: list[TrackObservation] = []

        for track_idx, det_idx in matches:
            track = self._tracks[track_idx]
            det = detections[det_idx]
            self._update_track(track, det, frame_idx)
            obs = self._emit_observation(track, frame_idx, matched=True)
            if obs:
                observations.append(obs)

        for idx in unmatched_tracks:
            track = self._tracks[idx]
            self._mark_missed(track, frame_idx)
            obs = self._emit_observation(track, frame_idx, matched=False)
            if obs:
                observations.append(obs)

        for det_idx in unmatched_detections:
            obs = self._start_new_track(detections[det_idx], frame_idx)
            if obs:
                observations.append(obs)

        # Cull dead tracks after emission to preserve indices during iteration
        survivors: list[ActiveTrack] = []
        for track in self._tracks:
            if track.state == TrackState.DEAD:
                self._retired_tracks.append(track)
            else:
                survivors.append(track)
        self._tracks = survivors
        self._prune_low_detection_rate()

        return observations

    def _prune_low_detection_rate(self) -> None:
        """Mark tracks as dead when detection rate falls below threshold."""
        threshold = float(getattr(self.params, "min_detection_rate", 0.0) or 0.0)
        if threshold <= 0.0:
            return
        survivors: list[ActiveTrack] = []
        for track in self._tracks:
            total = max(1, track.age)
            rate = track.hits / float(total)
            if rate < threshold:
                track.state = TrackState.DEAD
                self._retired_tracks.append(track)
            else:
                survivors.append(track)
        self._tracks = survivors

    # ------------------------------------------------------------------
    # Association helpers
    # ------------------------------------------------------------------
    def _associate(self, detections: Sequence[Detection]):
        return self._associate_subset(range(len(self._tracks)), detections)

    def _associate_subset(
        self,
        track_indices: Iterable[int],
        detections: Sequence[Detection],
        *,
        distance_gate: float | None = None,
    ):
        track_indices = list(track_indices)
        if not track_indices or not detections:
            return [], list(track_indices), list(range(len(detections)))

        gate = distance_gate if distance_gate is not None else self.params.distance_gate
        assignment_threshold = gate

        track_boxes = [self._tracks[idx].current_tlwh() for idx in track_indices]
        track_sizes = [self._tracks[idx].frame_size for idx in track_indices]
        det_boxes = [det.tlwh for det in detections]
        det_sizes = [det.frame_size for det in detections]

        distance_matrix = batched_center_distance(track_boxes, track_sizes, det_boxes, det_sizes)
        cost_matrix = np.full_like(distance_matrix, assignment_threshold + 1.0, dtype=float)

        matches = []
        unmatched_tracks = set(track_indices)
        unmatched_dets = set(range(len(detections)))

        valid_mask = (distance_matrix <= gate) & np.isfinite(distance_matrix)
        cost_matrix[valid_mask] = distance_matrix[valid_mask]

        row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind, strict=False):
            cost = cost_matrix[r, c]
            if cost > assignment_threshold:
                continue
            matches.append((track_indices[r], c))
            unmatched_tracks.discard(track_indices[r])
            unmatched_dets.discard(c)

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    # ------------------------------------------------------------------
    # Track lifecycle operations
    # ------------------------------------------------------------------
    def _predict_tracks(self) -> None:
        for track in self._tracks:
            track.mean, track.covariance = kf_predict(track.mean, track.covariance)
            track.age += 1
            track.misses += 1

    def _update_track(self, track: ActiveTrack, det: Detection, frame_idx: int) -> None:
        track.mean, track.covariance = kf_update(track.mean, track.covariance, det.tlwh)
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
        track.debug_color = None

    def _mark_missed(self, track: ActiveTrack, frame_idx: int) -> None:
        track.mask = None
        track.last_seen = frame_idx
        if track.misses > self.params.max_misses_M:
            track.state = TrackState.HIDDEN
            return
        track.smoothed_tlwh = self._smooth_tlwh(track, mean_to_tlwh(track.mean))

    def _start_new_track(self, det: Detection, frame_idx: int) -> TrackObservation | None:
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
        self._next_id += 1
        track.state = (
            TrackState.CONFIRMED
            if track.hits >= self.params.confirm_after_N
            else TrackState.TENTATIVE
        )
        track.smoothed_tlwh = self._smooth_tlwh(track, det.tlwh)
        self._tracks.append(track)
        return self._emit_observation(track, frame_idx, matched=True)

    def _emit_observation(
        self, track: ActiveTrack, frame_idx: int, *, matched: bool
    ) -> TrackObservation | None:
        if track.state == TrackState.DEAD:
            return None
        if track.state == TrackState.HIDDEN:
            return None

        tlwh = track.current_tlwh()
        tlwh = self._dilate_and_clip(tlwh, track.frame_size)

        should_blur = track.state != TrackState.DEAD

        observation = TrackObservation(
            frame=frame_idx,
            track_id=track.track_id,
            tlwh=tlwh,
            state=track.state,
            age=track.age,
            last_seen=track.last_seen,
            score=track.score,
            should_blur=should_blur,
            frame_size=track.frame_size,
            debug_color=track.debug_color,
            mask=track.mask,
        )
        track.history.append(observation)
        track.debug_color = None
        return observation

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _smooth_tlwh(self, track: ActiveTrack, tlwh: np.ndarray) -> np.ndarray:
        alpha = np.clip(self.params.temporal_smooth_alpha, 0.0, 1.0)
        if track.smoothed_tlwh is None:
            smoothed = tlwh.copy()
        else:
            smoothed = alpha * tlwh + (1.0 - alpha) * track.smoothed_tlwh
        return smoothed

    def _dilate_and_clip(self, tlwh: np.ndarray, frame_size: tuple[int, int] | None) -> np.ndarray:
        pct = max(0.0, self.params.bbox_dilate_pct)
        x, y, w, h = tlwh
        dw = w * pct
        dh = h * pct
        expanded = np.array([x - dw / 2, y - dh / 2, w + dw, h + dh], dtype=float)
        if frame_size:
            fw, fh = frame_size
            expanded[0] = float(np.clip(expanded[0], 0.0, fw))
            expanded[1] = float(np.clip(expanded[1], 0.0, fh))
            expanded[2] = float(np.clip(expanded[2], 0.0, fw - expanded[0]))
            expanded[3] = float(np.clip(expanded[3], 0.0, fh - expanded[1]))
        return expanded

    def _polars_to_detections(
        self, frame_idx: int, frame_df: pl.DataFrame
    ) -> tuple[list[Detection], list[Detection]]:
        confident: list[Detection] = []
        low_confident: list[Detection] = []
        width = None
        height = None
        if "frame_width" in frame_df.columns and "frame_height" in frame_df.columns:
            width = int(frame_df["frame_width"].item(0))
            height = int(frame_df["frame_height"].item(0))
        frame_size = (width, height) if width is not None and height is not None else None

        for row in frame_df.iter_rows(named=True):
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)
            score = float(row.get("confidence", row.get("score", 1.0)))
            is_confident = bool(row.get("is_confident", True))
            mask = row.get("mask")
            detection = Detection(
                frame_idx=frame_idx,
                tlwh=tlwh,
                score=score,
                frame_size=frame_size,
                is_confident=is_confident,
                mask=mask,
            )
            if is_confident:
                confident.append(detection)
            else:
                low_confident.append(detection)
        return confident, low_confident

    # ------------------------------------------------------------------
    # Accessors for post-processing
    # ------------------------------------------------------------------
    @property
    def track_history(self) -> list[TrackObservation]:
        return list(self._timeline)

    @property
    def finished_tracks(self) -> list[ActiveTrack]:
        return list(self._retired_tracks)

    def get_tracker_info(self) -> dict[str, object]:
        return {
            "tracker_type": type(self).__name__,
            "params": self.params.model_dump(),
        }


__all__ = ["ActiveTrack", "BaseTracker"]
