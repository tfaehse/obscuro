"""Bidirectional tracking wrapper that runs forward and backward passes."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from anonymizer.config import TrackerParams, TrackerType

from .base import BaseTracker
from .common import TrackObservation, TrackOrigin, TrackState

logger = logging.getLogger(__name__)


class BidirectionalTracker(BaseTracker):
    """
    Wrapper that runs tracking bidirectionally and merges results.

    This tracker runs a forward pass through the video, then a backward pass
    (running detections in reverse order), and merges both passes to improve
    tracking quality - especially for scenarios where objects start far/blurry
    and become close/clear (e.g., flying towards an object).

    The backward pass strengthens early tracks using later high-confidence
    detections, while the forward pass provides context for the end of the
    video.
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
        base_tracker_type: TrackerType = TrackerType.BYTETRACK,
    ) -> None:
        """
        Initialize bidirectional tracker.

        :param video_source: Path to video file (optional, for visual tracking)
        :param params: Tracker parameters for both passes
        :param cancel_event: Threading event for cancellation
        :param progress_callback: Callback for progress updates
        :param confidence_threshold: High confidence threshold
        :param low_score_threshold: Low confidence threshold
        :param base_tracker_type: Which tracker to use for forward/backward passes
        """
        # Initialize base tracker state
        super().__init__(
            video_source=video_source,
            params=params,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            confidence_threshold=confidence_threshold,
            low_score_threshold=low_score_threshold,
        )

        self.confidence_threshold = confidence_threshold
        self.low_score_threshold = low_score_threshold
        self.base_tracker_type = base_tracker_type

        # Prototype used for introspection; :meth:`track` builds a fresh instance
        # per pass so the forward and backward passes never share mutable state.
        self.base_tracker = self._create_base_tracker()

    def _create_base_tracker(self) -> BaseTracker:
        """Build a fresh base tracker instance (state isolated per pass)."""
        # Import here to avoid a circular dependency at module load time.
        from . import TRACKER_FACTORY, get_tracker

        tracker_name = self.base_tracker_type.value
        if tracker_name not in TRACKER_FACTORY:
            raise ValueError(f"Unknown tracker type: {self.base_tracker_type}")

        tracker_class = get_tracker(tracker_name)
        return tracker_class(
            video_source=self.video_source,
            params=self.params,
            cancel_event=self.cancel_event,
            progress_callback=self.progress_callback,
            confidence_threshold=self.confidence_threshold,
            low_score_threshold=self.low_score_threshold,
        )

    def track(self, detections: pl.DataFrame) -> pl.DataFrame:
        """
        Run bidirectional tracking and merge results.

        :param detections: DataFrame with detection data
        :return: Merged tracking results
        """
        if detections.is_empty():
            return self._empty_output()

        logger.info("Starting bidirectional tracking with %s", self.base_tracker_type.value)

        # Get total frame count for reversing
        max_frame = self._get_max_frame(detections)

        # Forward pass
        logger.info("Running forward pass...")
        forward_tracker = self._create_base_tracker()
        forward_df = forward_tracker.track(detections)
        forward_timeline = self._df_to_timeline(forward_df, origin=TrackOrigin.FORWARD_ONLY)

        # Backward pass (reverse detections) on a fresh tracker so it does not
        # inherit forward-pass state (active tracks, next id, ...).
        logger.info("Running backward pass...")
        reversed_detections = self._reverse_detections(detections, max_frame)
        backward_tracker = self._create_base_tracker()
        backward_df = backward_tracker.track(reversed_detections)

        # Restore backward results to original frame order
        backward_timeline = self._df_to_timeline(backward_df, origin=TrackOrigin.BACKWARD_ONLY)
        restored_backward = self._restore_backward_order(backward_timeline, max_frame)

        # Merge both passes
        logger.info("Merging forward and backward passes...")
        merged_timeline = self._merge_bidirectional(forward_timeline, restored_backward)
        self._timeline = merged_timeline

        # Convert to output DataFrame
        outputs = [obs.as_dict(include_debug=True) for obs in merged_timeline if obs.should_blur]
        return self._outputs_to_dataframe(outputs)

    def _get_max_frame(self, detections: pl.DataFrame) -> int:
        """Get the maximum frame number from detections."""
        if "frame" in detections.columns:
            return int(detections.get_column("frame").max())
        return detections.height - 1

    def _reverse_detections(self, detections: pl.DataFrame, max_frame: int) -> pl.DataFrame:
        """
        Reverse detections for backward pass.

        Inverts frame numbers so frame N becomes 0, N-1 becomes 1, etc.
        This allows the tracker to run backward through the video.
        """
        if detections.is_empty():
            return detections

        # Clone and reverse frame numbers
        reversed_df = detections.clone()
        if "frame" in reversed_df.columns:
            # Transform frame numbers: frame N becomes (max_frame - N)
            reversed_df = reversed_df.with_columns((max_frame - pl.col("frame")).alias("frame"))
            # Sort by new frame number (ascending) so processing goes 0, 1, 2, ...
            reversed_df = reversed_df.sort("frame")

        return reversed_df

    def _restore_backward_order(
        self, backward_timeline: list[TrackObservation], max_frame: int
    ) -> list[TrackObservation]:
        """
        Restore backward pass results to original frame numbers.

        After running backward pass on reversed frames, convert back to
        original frame indices (0 -> N, 1 -> N-1, etc.).
        """
        restored = []
        for obs in backward_timeline:
            # Create new observation with restored frame number
            restored_obs = TrackObservation(
                frame=max_frame - obs.frame,
                track_id=obs.track_id,  # Will be remapped during merge
                tlwh=obs.tlwh.copy(),
                state=obs.state,
                age=obs.age,
                last_seen=max_frame - obs.last_seen,
                score=obs.score,
                should_blur=obs.should_blur,
                frame_size=obs.frame_size,
                debug_color=obs.debug_color,
                mask=obs.mask,
            )
            restored.append(restored_obs)
        return restored

    def _df_to_timeline(
        self, df: pl.DataFrame, origin: TrackOrigin = TrackOrigin.FORWARD_ONLY
    ) -> list[TrackObservation]:
        """Convert DataFrame to list of TrackObservations."""
        timeline = []
        if df.is_empty():
            return timeline

        required_cols = {
            "frame",
            "track_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "state",
            "age",
            "last_seen",
            "score",
        }
        if not required_cols.issubset(set(df.columns)):
            logger.warning("DataFrame missing required columns for timeline conversion")
            return timeline

        for row in df.iter_rows(named=True):
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)

            state = TrackState.TENTATIVE
            if isinstance(row["state"], str):
                try:
                    state = TrackState(row["state"].lower())
                except ValueError:
                    state = TrackState.TENTATIVE

            should_blur = state != TrackState.TENTATIVE

            obs = TrackObservation(
                frame=int(row["frame"]),
                track_id=int(row["track_id"]),
                tlwh=tlwh,
                state=state,
                age=int(row.get("age", 0)),
                last_seen=int(row.get("last_seen", row["frame"])),
                score=float(row.get("score", 0.5)),
                should_blur=should_blur,
                frame_size=None,
                debug_color=None,
                mask=row.get("mask"),
                origin=origin,
            )
            timeline.append(obs)

        return timeline

    def _merge_bidirectional(
        self,
        forward_timeline: list[TrackObservation],
        backward_timeline: list[TrackObservation],
    ) -> list[TrackObservation]:
        """
        Merge forward and backward tracking results into stable trajectories.

        Matching happens at the *track* level: each forward track is paired with
        the backward track that shares the highest average IoU across the frames
        where both are present. A paired track pair receives a single merged id,
        so an object keeps the same id across every frame it appears in. Tracks
        that only exist in one pass keep their own id.

        :param forward_timeline: Track observations from forward pass
        :param backward_timeline: Track observations from backward pass (restored)
        :return: Merged timeline with consistent track IDs
        """
        forward_tracks = self._group_by_track(forward_timeline)
        backward_tracks = self._group_by_track(backward_timeline)

        # Pair forward<->backward tracks, then assign one merged id per object.
        paired_backward: set[int] = set()
        forward_id_map: dict[int, int] = {}
        backward_id_map: dict[int, int] = {}
        next_merged_id = 1

        for forward_id, forward_obs in forward_tracks.items():
            best_backward_id = self._best_backward_match(
                forward_obs, backward_tracks, exclude=paired_backward
            )
            forward_id_map[forward_id] = next_merged_id
            if best_backward_id is not None:
                paired_backward.add(best_backward_id)
                backward_id_map[best_backward_id] = next_merged_id
            next_merged_id += 1

        # Backward-only tracks get their own ids.
        for backward_id in backward_tracks:
            if backward_id not in backward_id_map:
                backward_id_map[backward_id] = next_merged_id
                next_merged_id += 1

        # Group observations by frame and merge frame-by-frame using the maps.
        forward_by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
        backward_by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
        for obs in forward_timeline:
            forward_by_frame[obs.frame].append(obs)
        for obs in backward_timeline:
            backward_by_frame[obs.frame].append(obs)

        merged: list[TrackObservation] = []
        for frame_num in sorted(set(forward_by_frame) | set(backward_by_frame)):
            merged.extend(
                self._merge_frame_observations(
                    forward_by_frame.get(frame_num, []),
                    backward_by_frame.get(frame_num, []),
                    forward_id_map,
                    backward_id_map,
                )
            )

        merged.sort(key=lambda o: (o.frame, o.track_id))

        logger.info(
            "Merged %d forward + %d backward observations into %d total",
            len(forward_timeline),
            len(backward_timeline),
            len(merged),
        )

        return merged

    @staticmethod
    def _group_by_track(
        timeline: list[TrackObservation],
    ) -> dict[int, list[TrackObservation]]:
        """Group observations by their source track id."""
        grouped: dict[int, list[TrackObservation]] = defaultdict(list)
        for obs in timeline:
            grouped[obs.track_id].append(obs)
        return grouped

    def _best_backward_match(
        self,
        forward_obs: list[TrackObservation],
        backward_tracks: dict[int, list[TrackObservation]],
        *,
        exclude: set[int],
    ) -> int | None:
        """Return the backward track id with the highest average IoU to ``forward_obs``."""
        threshold = float(self.params.bidirectional_merge_iou_threshold)
        forward_by_frame = {obs.frame: obs for obs in forward_obs}
        best_id: int | None = None
        best_iou = threshold
        for backward_id, backward_obs in backward_tracks.items():
            if backward_id in exclude:
                continue
            backward_by_frame = {obs.frame: obs for obs in backward_obs}
            shared_frames = set(forward_by_frame) & set(backward_by_frame)
            if not shared_frames:
                continue
            total_iou = 0.0
            for frame_num in shared_frames:
                total_iou += self._calculate_iou(
                    forward_by_frame[frame_num].tlwh,
                    backward_by_frame[frame_num].tlwh,
                )
            average_iou = total_iou / len(shared_frames)
            if average_iou > best_iou:
                best_iou = average_iou
                best_id = backward_id
        return best_id

    def _should_merge_observations(
        self, forward_obs: TrackObservation, backward_obs: TrackObservation
    ) -> bool:
        """
        Per-frame heuristic: do these two observations likely belong to the same
        object? Uses IoU against the configured merge threshold.

        Track-level matching in :meth:`_merge_bidirectional` is what assigns
        stable ids; this helper supports single-frame decisions and tests.
        """
        threshold = float(self.params.bidirectional_merge_iou_threshold)
        iou = self._calculate_iou(forward_obs.tlwh, backward_obs.tlwh)
        return bool(iou > threshold)

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Boxes are in [x, y, w, h] format.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _merge_frame_observations(
        self,
        forward_obs: list[TrackObservation],
        backward_obs: list[TrackObservation],
        forward_id_map: dict[int, int],
        backward_id_map: dict[int, int],
    ) -> list[TrackObservation]:
        """
        Merge observations for a single frame using the global id maps.

        A forward and backward observation that share a merged id are combined
        into one observation (preferring the backward box, which comes from
        later, higher-confidence frames). Unmatched observations pass through
        with their original origin preserved.
        """
        weight = float(self.params.bidirectional_confidence_weight)
        merged: list[TrackObservation] = []
        used_backward: set[int] = set()

        backward_by_merged: dict[int, list[TrackObservation]] = defaultdict(list)
        for b_obs in backward_obs:
            merged_id = backward_id_map.get(b_obs.track_id, b_obs.track_id)
            backward_by_merged[merged_id].append(b_obs)

        for f_obs in forward_obs:
            f_merged_id = forward_id_map.get(f_obs.track_id, f_obs.track_id)
            match = None
            for b_obs in backward_by_merged.get(f_merged_id, []):
                if b_obs.track_id in used_backward:
                    continue
                match = b_obs
                break

            if match is not None:
                used_backward.add(match.track_id)
                merged.append(self._combine(f_obs, match, f_merged_id, weight))
            else:
                merged.append(replace(f_obs, track_id=f_merged_id))

        for b_obs in backward_obs:
            if b_obs.track_id in used_backward:
                continue
            b_merged_id = backward_id_map.get(b_obs.track_id, b_obs.track_id)
            merged.append(replace(b_obs, track_id=b_merged_id))

        return merged

    def _combine(
        self,
        forward_obs: TrackObservation,
        backward_obs: TrackObservation,
        merged_id: int,
        weight: float,
    ) -> TrackObservation:
        """Combine a matched forward/backward pair into a single observation."""
        # Prefer the backward box (later frames tend to carry better detections)
        # and blend the confidence scores, with ``weight`` favoring backward.
        score = (1.0 - weight) * forward_obs.score + weight * backward_obs.score
        return TrackObservation(
            frame=forward_obs.frame,
            track_id=merged_id,
            tlwh=backward_obs.tlwh.copy(),
            state=max(forward_obs.state, backward_obs.state, key=lambda s: s.value),
            age=max(forward_obs.age, backward_obs.age),
            last_seen=forward_obs.frame,
            score=float(score),
            should_blur=forward_obs.should_blur or backward_obs.should_blur,
            frame_size=forward_obs.frame_size,
            debug_color=forward_obs.debug_color,
            mask=backward_obs.mask if backward_obs.mask is not None else forward_obs.mask,
            origin=TrackOrigin.MERGED,
        )

    def _empty_output(self) -> pl.DataFrame:
        """Return empty DataFrame with correct schema."""
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
                "origin": pl.Utf8,
            }
        )

    def _outputs_to_dataframe(self, outputs: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert output dictionaries to Polars DataFrame."""
        if not outputs:
            return self._empty_output()

        df = pl.DataFrame(outputs)
        if "mask" in df.columns:
            df = df.with_columns(pl.col("mask").cast(pl.Int64, strict=False))
        return df

    @property
    def track_history(self) -> list[TrackObservation]:
        """Return full tracking timeline (for offline linking)."""
        return self._timeline.copy()

    def clear(self) -> None:
        """Clear tracker state."""
        self._timeline.clear()
        if hasattr(self.base_tracker, "clear"):
            self.base_tracker.clear()
