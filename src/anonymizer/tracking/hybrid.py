from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from anonymizer.config import TrackerParams
from anonymizer.utils.geometry import iou_tlwh

from .base import ActiveTrack, BaseTracker
from .common import Detection, TrackObservation, normalized_center
from .kalman import mean_to_tlwh
from .kalman import update as kf_update


def _create_visual_tracker(backend: str) -> cv2.Tracker | None:
    backend = backend.lower()
    constructors = []
    if backend == "csrt":
        constructors.extend(
            [
                getattr(cv2, "TrackerCSRT_create", None),
                getattr(cv2, "legacy_TrackerCSRT", None),
            ]
        )
    elif backend == "kcf":
        constructors.extend(
            [
                getattr(cv2, "TrackerKCF_create", None),
                getattr(cv2, "legacy_TrackerKCF", None),
            ]
        )
    elif backend in {"siam", "siamrpn"}:
        constructors.append(getattr(cv2, "TrackerMIL_create", None))
    else:
        constructors.extend(
            [
                getattr(cv2, "TrackerCSRT_create", None),
                getattr(cv2, "legacy_TrackerCSRT", None),
            ]
        )

    for ctor in constructors:
        if callable(ctor):
            with contextlib.suppress(Exception):
                tracker = ctor()
                return tracker
    return None


class HybridSOTTracker(BaseTracker):
    """SORT augmented with a per-track visual tracker to bridge detector gaps."""

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
        self._cv_capture = None
        self._next_frame_to_grab = 0
        self._last_frame: np.ndarray | None = None
        self._current_low_conf: list[Detection] = []

    def reconfigure(self, params: TrackerParams) -> None:
        super().reconfigure(params)
        self._release_capture()

    def _release_capture(self) -> None:
        if self._cv_capture is not None:
            with contextlib.suppress(Exception):
                self._cv_capture.release()
        self._cv_capture = None
        self._last_frame = None
        self._next_frame_to_grab = 0

    def _ensure_capture(self) -> None:
        if self._cv_capture is None and self.video_source is not None:
            self._cv_capture = cv2.VideoCapture(str(self.video_source))
            self._next_frame_to_grab = 0

    def _get_frame(self, frame_idx: int) -> np.ndarray | None:
        if self.video_source is None:
            return None
        self._ensure_capture()
        if self._cv_capture is None:
            return None
        if frame_idx < self._next_frame_to_grab:
            return self._last_frame
        while self._next_frame_to_grab <= frame_idx:
            ret, frame = self._cv_capture.read()
            if not ret:
                return None
            self._last_frame = frame
            self._next_frame_to_grab += 1
        return self._last_frame

    def _update_track(self, track: ActiveTrack, det: Detection, frame_idx: int) -> None:
        super()._update_track(track, det, frame_idx)
        if not self.params.use_visual_tracker:
            return
        frame = self._get_frame(frame_idx)
        if frame is None:
            return
        tracker = _create_visual_tracker(self.params.vt_backend)
        if tracker is None:
            return
        tlwh = det.tlwh
        bbox = (float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3]))
        try:
            tracker.init(frame, bbox)
            track.visual_tracker = tracker
        except Exception:
            track.visual_tracker = None

    def _mark_missed(self, track: ActiveTrack, frame_idx: int) -> None:
        if (
            self.params.use_visual_tracker
            and track.visual_tracker is not None
            and track.misses <= self.params.vt_max_age
        ):
            frame = self._get_frame(frame_idx)
            if frame is not None:
                ok, bbox = track.visual_tracker.update(frame)
                if ok:
                    tlwh = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=float)
                    if self._within_drift_gate(track, tlwh):
                        if self._promote_low_conf_candidate(track, frame_idx, tlwh):
                            return
                        track.mean, track.covariance = kf_update(track.mean, track.covariance, tlwh)
                        track.smoothed_tlwh = self._smooth_tlwh(track, mean_to_tlwh(track.mean))
                        track.score *= 0.9
                        track.last_seen = frame_idx
                        track.misses = 0
                        return
        super()._mark_missed(track, frame_idx)

    def _process_frame(
        self,
        frame_idx: int,
        detections: Sequence[Detection],
        low_conf_detections: Sequence[Detection] | None = None,
    ) -> list[TrackObservation]:
        self._current_low_conf = list(low_conf_detections or [])
        return super()._process_frame(frame_idx, detections, low_conf_detections)

    def _promote_low_conf_candidate(
        self, track: ActiveTrack, frame_idx: int, tlwh: np.ndarray
    ) -> bool:
        if not self._current_low_conf:
            return False
        best_det: Detection | None = None
        best_iou = 0.0
        for det in list(self._current_low_conf):
            iou = iou_tlwh(tlwh, det.tlwh)
            if iou > 0.1 and iou > best_iou:
                best_iou = iou
                best_det = det
        if best_det is None:
            return False
        with contextlib.suppress(ValueError):
            self._current_low_conf.remove(best_det)
        self._update_track(track, best_det, frame_idx)
        track.debug_color = (0, 255, 0)
        return True

    def _within_drift_gate(self, track: ActiveTrack, tlwh: np.ndarray) -> bool:
        if self.params.drift_gate <= 0:
            return True
        frame_size = track.frame_size
        if not frame_size or frame_size[0] <= 0 or frame_size[1] <= 0:
            return True
        predicted = track.current_tlwh()
        pred_center = normalized_center(predicted, frame_size)
        det_center = normalized_center(tlwh, frame_size)
        if pred_center is None or det_center is None:
            return True
        drift = float(np.linalg.norm(det_center - pred_center))
        return drift <= self.params.drift_gate

    def get_tracker_info(self) -> dict[str, object]:
        info = super().get_tracker_info()
        info["visual_tracker"] = self.params.vt_backend
        return info

    def __del__(self) -> None:  # pragma: no cover - defensive
        self._release_capture()


__all__ = ["HybridSOTTracker"]
