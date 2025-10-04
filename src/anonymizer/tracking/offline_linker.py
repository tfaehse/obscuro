from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

from anonymizer.config import TrackerParams

from .base import BaseTracker
from .common import TrackObservation, TrackState, center_distance, normalized_center

MIN_SEGMENT_LENGTH = 10
MAX_GAP_FRAMES = 30
PER_FRAME_ERROR_LIMIT = 0.05
SMOOTHNESS_LIMIT = 0.02
VELOCITY_COS_LIMIT = 0.8
VELOCITY_RATIO_LIMIT = 3.0


def _trajectory_alignment_error(
    prev_summary: dict,
    next_summary: dict,
    params: TrackerParams,
) -> tuple[float, float] | None:
    centers_prev: np.ndarray = prev_summary["centers"]
    centers_next: np.ndarray = next_summary["centers"]

    if centers_prev.size == 0 or centers_next.size == 0:
        return None

    last_center = centers_prev[-1]
    first_center = centers_next[0]

    velocity_prev = _compute_velocity(prev_summary, tail=True)
    velocity_next = _compute_velocity(next_summary, tail=False)

    norm_prev = float(np.linalg.norm(velocity_prev)) if velocity_prev is not None else 0.0
    norm_next = float(np.linalg.norm(velocity_next)) if velocity_next is not None else 0.0

    if norm_prev < 1e-6 and norm_next < 1e-6:
        pass
    elif norm_prev < 1e-6 or norm_next < 1e-6:
        return None
    else:
        dot = float(np.dot(velocity_prev, velocity_next) / (norm_prev * norm_next))
        if dot < VELOCITY_COS_LIMIT:
            return None
        ratio = max(norm_prev, norm_next) / max(min(norm_prev, norm_next), 1e-6)
        if ratio > VELOCITY_RATIO_LIMIT:
            return None

    gap = int(next_summary["start_frame"] - prev_summary["end_frame"])
    if gap <= 0:
        return None

    forward_steps = min(centers_next.shape[0], 5)
    backward_steps = min(centers_prev.shape[0], 5)

    forward_errors: list[float] = []
    if norm_prev < 1e-6:
        for k in range(forward_steps):
            predicted = last_center
            delta = float(np.linalg.norm(predicted - centers_next[k]))
            forward_errors.append(delta / max(gap + k + 1, 1))
    else:
        for k in range(forward_steps):
            predicted = last_center + velocity_prev * (gap + k + 1)
            delta = float(np.linalg.norm(predicted - centers_next[k]))
            forward_errors.append(delta / max(gap + k + 1, 1))

    backward_errors: list[float] = []
    if norm_next < 1e-6:
        for k in range(backward_steps):
            predicted = first_center
            delta = float(np.linalg.norm(predicted - centers_prev[-(k + 1)]))
            backward_errors.append(delta / max(k + 1, 1))
    else:
        for k in range(backward_steps):
            predicted = first_center - velocity_next * (k + 1)
            delta = float(np.linalg.norm(predicted - centers_prev[-(k + 1)]))
            backward_errors.append(delta / max(k + 1, 1))

    if not forward_errors:
        return None

    mean_error = float(np.mean(forward_errors))
    smooth_measure = max((forward_errors + backward_errors) or forward_errors)

    if smooth_measure > 1.0:
        smooth_measure = float("inf")

    smooth_residual = _evaluate_gap_smoothness(prev_summary, next_summary)
    if smooth_residual is None:
        smooth_residual = smooth_measure
    else:
        smooth_residual = max(smooth_residual, smooth_measure)

    return mean_error, smooth_residual


def _make_summary(observations: Iterable[TrackObservation], track_ids: Iterable[int]) -> dict:
    ordered = sorted(observations, key=lambda o: o.frame)
    frames: list[float] = []
    centers: list[np.ndarray] = []
    for obs in ordered:
        center = normalized_center(obs.tlwh, obs.frame_size)
        if center is None:
            continue
        frames.append(float(obs.frame))
        centers.append(center.astype(float))
    frames_arr = np.asarray(frames, dtype=float) if frames else np.empty((0,), dtype=float)
    centers_arr = np.asarray(centers, dtype=float) if centers else np.empty((0, 2), dtype=float)
    return {
        "track_ids": set(track_ids),
        "observations": ordered,
        "frames": frames_arr,
        "centers": centers_arr,
        "start_frame": ordered[0].frame if ordered else None,
        "end_frame": ordered[-1].frame if ordered else None,
        "length": len(ordered),
        "eligible": len(ordered) >= MIN_SEGMENT_LENGTH,
    }


def _clone_observation(obs: TrackObservation, new_track_id: int) -> TrackObservation:
    return TrackObservation(
        frame=obs.frame,
        track_id=new_track_id,
        tlwh=obs.tlwh.copy(),
        state=obs.state,
        age=obs.age,
        last_seen=obs.last_seen,
        score=obs.score,
        should_blur=obs.should_blur,
        frame_size=obs.frame_size,
        debug_color=obs.debug_color,
    )


def _interpolate_gap(
    obs_a: TrackObservation,
    obs_b: TrackObservation,
    track_id: int,
) -> list[TrackObservation]:
    gap = obs_b.frame - obs_a.frame - 1
    if gap <= 0:
        return []

    start = obs_a.tlwh.astype(float)
    end = obs_b.tlwh.astype(float)
    start_score = float(obs_a.score)
    end_score = float(obs_b.score)
    result: list[TrackObservation] = []
    for step in range(1, gap + 1):
        alpha = step / (gap + 1)
        tlwh = (1.0 - alpha) * start + alpha * end
        score = (1.0 - alpha) * start_score + alpha * end_score
        frame_idx = obs_a.frame + step
        frame_size = obs_a.frame_size or obs_b.frame_size
        result.append(
            TrackObservation(
                frame=frame_idx,
                track_id=track_id,
                tlwh=tlwh.astype(float),
                state=TrackState.CONFIRMED,
                age=0,
                last_seen=frame_idx,
                score=score,
                should_blur=False,
                frame_size=frame_size,
                debug_color=(255, 255, 0),
            )
        )
    return result


def _polyfit_residual(frames: np.ndarray, values: np.ndarray, degree: int = 2) -> float:
    if frames.size == 0:
        return float("inf")
    deg = min(degree, max(int(frames.size) - 1, 1))
    try:
        coeffs = np.polyfit(frames, values, deg)
    except np.linalg.LinAlgError:
        return float("inf")
    fitted = np.polyval(coeffs, frames)
    return float(np.sqrt(np.mean((values - fitted) ** 2)))


def _evaluate_gap_smoothness(prev_summary: dict, next_summary: dict) -> float | None:
    frames_prev: np.ndarray = prev_summary["frames"]
    centers_prev: np.ndarray = prev_summary["centers"]
    frames_next: np.ndarray = next_summary["frames"]
    centers_next: np.ndarray = next_summary["centers"]

    if centers_prev.size == 0 or centers_next.size == 0:
        return None

    tail_size = 6
    frames_prev_tail = frames_prev[-tail_size:]
    centers_prev_tail = centers_prev[-tail_size:]
    frames_next_head = frames_next[:tail_size]
    centers_next_head = centers_next[:tail_size]

    last_frame = prev_summary.get("end_frame")
    first_frame = next_summary.get("start_frame")
    if last_frame is None or first_frame is None:
        return None
    last_center = centers_prev_tail[-1]
    first_center = centers_next_head[0]

    gap = int(first_frame - last_frame - 1)
    gap_frames: list[float] = []
    gap_centers: list[np.ndarray] = []
    if gap > 0:
        for step in range(1, gap + 1):
            alpha = step / (gap + 1)
            gap_frames.append(float(last_frame + step))
            gap_centers.append(((1.0 - alpha) * last_center + alpha * first_center).astype(float))

    combined_frames = np.concatenate(
        (
            frames_prev_tail,
            np.asarray(gap_frames, dtype=float) if gap_frames else np.empty((0,), dtype=float),
            frames_next_head,
        )
    )
    combined_centers = np.vstack(
        (
            centers_prev_tail,
            np.asarray(gap_centers, dtype=float) if gap_centers else np.empty((0, 2), dtype=float),
            centers_next_head,
        )
    )

    if combined_frames.size < 3:
        return None

    residual_x = _polyfit_residual(combined_frames, combined_centers[:, 0])
    residual_y = _polyfit_residual(combined_frames, combined_centers[:, 1])
    return max(residual_x, residual_y)


def _compute_velocity(summary: dict, *, tail: bool) -> np.ndarray | None:
    frames: np.ndarray = summary["frames"]
    centers: np.ndarray = summary["centers"]
    if frames.size < 2:
        return None
    window = min(5, frames.size - 1)
    if tail:
        idx_end = frames.size - 1
        idx_start = max(0, idx_end - window)
    else:
        idx_start = 0
        idx_end = min(window, frames.size - 1)
    dt = frames[idx_end] - frames[idx_start]
    if np.isclose(dt, 0.0):
        return None
    velocity = (centers[idx_end] - centers[idx_start]) / dt
    return velocity


def _generate_filled_track(summary: dict, new_track_id: int) -> list[TrackObservation]:
    observations = sorted(summary["observations"], key=lambda o: o.frame)
    filled: list[TrackObservation] = []
    for idx, obs in enumerate(observations):
        cloned = _clone_observation(obs, new_track_id)
        filled.append(cloned)
        if idx + 1 >= len(observations):
            continue
        next_obs = observations[idx + 1]
        filled.extend(_interpolate_gap(obs, next_obs, new_track_id))
    return filled


def link_tracklets(
    video_id: str | None,
    tracklets: Iterable[TrackObservation],
    params: TrackerParams,
) -> tuple[dict[int, int], dict[int, list[TrackObservation]]]:
    """Link fragmented tracklets and return ID mapping plus gap-filled trajectories."""
    grouped: dict[int, list[TrackObservation]] = defaultdict(list)
    for obs in tracklets:
        grouped[obs.track_id].append(obs)

    summaries: list[dict] = []
    for track_id, observations in grouped.items():
        if not observations:
            continue
        summary = _make_summary(observations, {track_id})
        summaries.append(summary)

    summaries.sort(key=lambda item: item["start_frame"] or 0)
    configured_gate = getattr(params, "offline_linker_per_frame_gate", None)
    if configured_gate is None:
        configured_gate = max(params.distance_gate, 0.02)
    per_frame_gate = min(configured_gate, PER_FRAME_ERROR_LIMIT)
    smoothness_gate = SMOOTHNESS_LIMIT
    max_gap = min(MAX_GAP_FRAMES, getattr(params, "offline_linker_max_misses", MAX_GAP_FRAMES))

    if not summaries:
        return {}, {}

    while True:
        best_candidate: tuple[float, float, int, int] | None = None
        for i, prev in enumerate(summaries):
            if not prev.get("eligible", False):
                continue
            end_prev = prev.get("end_frame")
            if end_prev is None:
                continue
            prev_last_obs: TrackObservation = prev["observations"][-1]
            for j in range(i + 1, len(summaries)):
                nxt = summaries[j]
                if not nxt.get("eligible", False):
                    continue
                start_next = nxt.get("start_frame")
                if start_next is None:
                    continue
                frame_gap = start_next - end_prev
                if frame_gap <= 0 or frame_gap > max_gap:
                    continue
                first_obs: TrackObservation = nxt["observations"][0]
                distance = center_distance(
                    prev_last_obs.tlwh,
                    prev_last_obs.frame_size,
                    first_obs.tlwh,
                    first_obs.frame_size,
                )
                if not np.isfinite(distance) or distance > params.distance_gate:
                    continue
                result = _trajectory_alignment_error(prev, nxt, params)
                if result is None:
                    continue
                mean_error, smoothness = result
                per_frame_error = mean_error
                if per_frame_error > per_frame_gate:
                    continue
                if smoothness > smoothness_gate:
                    continue
                if best_candidate is None or (
                    per_frame_error < best_candidate[0]
                    or (
                        np.isclose(per_frame_error, best_candidate[0])
                        and smoothness < best_candidate[1]
                    )
                ):
                    best_candidate = (per_frame_error, smoothness, i, j)

        if best_candidate is None:
            break

        _, _, prev_idx, next_idx = best_candidate
        prev_summary = summaries[prev_idx]
        next_summary = summaries[next_idx]
        combined_observations = prev_summary["observations"] + next_summary["observations"]
        combined_ids = prev_summary["track_ids"].union(next_summary["track_ids"])
        merged_summary = _make_summary(combined_observations, combined_ids)
        summaries[prev_idx] = merged_summary
        summaries.pop(next_idx)
        summaries.sort(key=lambda item: item["start_frame"] or 0)

    mapping: dict[int, int] = {}
    filled_tracks: dict[int, list[TrackObservation]] = {}
    next_id = 1
    for summary in sorted(summaries, key=lambda item: item["start_frame"] or 0):
        new_id = next_id
        next_id += 1
        for original_id in summary["track_ids"]:
            mapping[int(original_id)] = new_id
        filled_tracks[new_id] = _generate_filled_track(summary, new_id)

    return mapping, filled_tracks


class OfflineLinkerTracker(BaseTracker):
    """Online tracker followed by an offline linking pass to merge fragments."""

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
        self.video_id = str(video_source) if video_source else None
        self.offline_linked_tracks: dict[int, list[TrackObservation]] | None = None

    def track(self, detections: pl.DataFrame) -> pl.DataFrame:
        df = super().track(detections)
        if df.is_empty():
            return df

        mapping, filled_tracks = link_tracklets(self.video_id, self.track_history, self.params)
        self.offline_linked_tracks = filled_tracks
        if not mapping:
            return df

        remapped = df.with_columns(
            pl.col("track_id")
            .map_elements(lambda tid: mapping.get(int(tid), int(tid)))
            .alias("track_id")
        )
        return remapped


__all__ = ["OfflineLinkerTracker", "link_tracklets"]
