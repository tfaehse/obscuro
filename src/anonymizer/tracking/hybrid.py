from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from anonymizer.config import TrackerParams
from anonymizer.paths import (
    get_detection_models_dir,
    get_models_dir,
    get_tracking_models_dir,
)
from anonymizer.tracking.embeddings import get_embedding_model
from anonymizer.tracking.fused import FusedTracker
from anonymizer.tracking.utils import (
    clamp_bbox,
    cosine_similarities,
    crop_patch,
    update_weighted_embedding,
)
from anonymizer.utils.geometry import iou_tlwh

from .base import ActiveTrack
from .common import Detection, TrackObservation, normalized_center
from .kalman import mean_to_tlwh
from .kalman import update as kf_update

logger = logging.getLogger("obscuro.tracking.hybrid")

_TRACKERNANO_FILENAMES = {
    "backbone": (
        "trackernano_backbone.onnx",
        "tracker_nano_backbone.onnx",
        "backbone.onnx",
        "nanotrack_backbone.onnx",
    ),
    "neckhead": (
        "trackernano_neckhead.onnx",
        "tracker_nano_neckhead.onnx",
        "neckhead.onnx",
        "nanotrack_head.onnx",
    ),
}


def _create_visual_tracker(backend: str) -> cv2.Tracker | None:
    backend = backend.lower()
    if backend in {"trackernano", "nano"}:
        return _create_trackernano_tracker()

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


def _create_trackernano_tracker() -> cv2.Tracker | None:
    params_ctor = getattr(cv2, "TrackerNano_Params", None)
    tracker_ctor = getattr(cv2, "TrackerNano_create", None)
    if not params_ctor or not tracker_ctor:
        logger.warning("TrackerNano backend is unavailable in the current OpenCV build.")
        return None

    weights: dict[str, Path] = {}
    for key in ("backbone", "neckhead"):
        resolved = _resolve_trackernano_weight(key)
        if not resolved:
            logger.warning(
                "Missing TrackerNano %s weights. Expected one of %s in %s or %s.",
                key,
                _TRACKERNANO_FILENAMES[key],
                get_tracking_models_dir(create=False),
                Path(__file__).resolve().parents[2] / "models" / "tracking",
            )
            return None
        weights[key] = resolved

    params = params_ctor()
    params.backbone = str(weights["backbone"])
    params.neckhead = str(weights["neckhead"])
    try:
        return tracker_ctor(params)
    except Exception:
        logger.exception("Failed to initialize TrackerNano backend.")
        return None


def _resolve_trackernano_weight(kind: str) -> Path | None:
    search_roots: list[Path] = []
    with contextlib.suppress(Exception):
        search_roots.extend(
            [
                get_tracking_models_dir(create=False),
                get_detection_models_dir(create=False),
                get_models_dir(create=False),
            ]
        )

    repo_root = Path(__file__).resolve().parents[3] / "models"
    search_roots.append(repo_root / "tracking")
    search_roots.append(repo_root / "detection")
    search_roots.append(repo_root)
    search_roots.append(Path.cwd())

    seen: set[Path] = set()
    for root in search_roots:
        if not root or not root.exists():
            continue
        if root in seen:
            continue
        seen.add(root)
        for filename in _TRACKERNANO_FILENAMES.get(kind, ()):
            candidate = root / filename
            if candidate.is_file():
                return candidate


class HybridSOTTracker(FusedTracker):
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
        self._current_low_conf: list[Detection] = []
        self._embedding_model = get_embedding_model()

    @staticmethod
    def _describe_frame(frame: np.ndarray | None) -> str:
        if frame is None:
            return "frame=None"
        contiguous = bool(frame.flags["C_CONTIGUOUS"])
        return f"shape={tuple(frame.shape)}, dtype={frame.dtype}, contiguous={contiguous}"

    @staticmethod
    def _inflate_bbox(
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int, int] | tuple[int, int],
        scale: float = 1.2,
    ) -> tuple[int, int, int, int]:
        """Expand a bbox about its center by `scale`, clamped to the frame."""
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        new_w = w * scale
        new_h = h * scale
        new_x = round(cx - new_w / 2.0)
        new_y = round(cy - new_h / 2.0)
        enlarged = (new_x, new_y, round(new_w), round(new_h))
        clamped = clamp_bbox(enlarged, frame_shape)
        return clamped if clamped is not None else bbox

    def _run_visual_tracker_update(
        self, track: ActiveTrack, frame: np.ndarray | None
    ) -> tuple[bool, tuple[float, float, float, float] | None]:
        if frame is None or track.visual_tracker is None:
            return False, None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "VT update: track=%s frame=%s",
                track.track_id,
                self._describe_frame(frame),
            )
        try:
            return track.visual_tracker.update(frame)
        except cv2.error as exc:
            logger.warning(
                "Visual tracker update failed for track %s (%s): %s",
                track.track_id,
                self._describe_frame(frame),
                exc,
            )
        except Exception:
            logger.exception(
                "Unexpected visual tracker failure for track %s (%s)",
                track.track_id,
                self._describe_frame(frame),
            )
        track.visual_tracker = None
        return False, None

    # Association inherited from FusedTracker

    def _update_track(self, track: ActiveTrack, det: Detection, frame_idx: int) -> None:
        super()._update_track(track, det, frame_idx)
        if not self.params.use_visual_tracker:
            return
        frame = self._prepare_frame(self._get_frame(frame_idx))
        if frame is None:
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "VT init candidate: track=%s frame_idx=%s frame=%s det_tlwh=%s",
                track.track_id,
                frame_idx,
                self._describe_frame(frame),
                det.tlwh,
            )
        tracker = _create_visual_tracker(self.params.vt_backend)
        if tracker is None:
            return
        tlwh = np.asarray(det.tlwh, dtype=float).reshape(-1)
        x, y, w, h = (round(v) for v in tlwh[:4])
        bbox = (x, y, max(1, w), max(1, h))
        bbox = self._inflate_bbox(bbox, frame.shape, scale=1.2)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "VT init bbox: track=%s frame_idx=%s bbox=%s frame_shape=%s backend=%s",
                track.track_id,
                frame_idx,
                bbox,
                frame.shape if frame is not None else None,
                self.params.vt_backend,
            )
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "VT init call: track=%s feeding bbox=%s into %s tracker",
                    track.track_id,
                    bbox,
                    self.params.vt_backend,
                )
            tracker.init(frame, bbox)
            track.visual_tracker = tracker
            if self._embedding_model is not None:
                patch = crop_patch(frame, bbox)
                if patch is not None:
                    emb = self._embedding_model.embed(patch)
                    track.vt_embeddings = [*track.vt_embeddings, emb][-3:]
                    track.vt_embedding_rep = update_weighted_embedding(
                        track.vt_embedding_rep, emb, det.score
                    )
        except Exception as e:
            logger.debug(f"Failed to initialize visual tracker: {e}")
            track.visual_tracker = None

    def _mark_missed(self, track: ActiveTrack, frame_idx: int) -> None:
        if (
            self.params.use_visual_tracker
            and track.visual_tracker is not None
            and track.misses <= self.params.vt_max_age
        ):
            frame = self._prepare_frame(self._get_frame(frame_idx))
            if frame is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "VT miss update: track=%s frame_idx=%s frame=%s",
                        track.track_id,
                        frame_idx,
                        self._describe_frame(frame),
                    )
                ok, bbox = self._run_visual_tracker_update(track, frame)
                if ok:
                    if self._embedding_model is not None:
                        with contextlib.suppress(Exception):
                            patch = crop_patch(
                                frame, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                            )
                            emb = self._embedding_model.embed(patch if patch is not None else frame)
                            history = track.vt_embeddings
                            rep = track.vt_embedding_rep
                            if rep is not None:
                                sim_rep = float(
                                    np.dot(
                                        rep / (np.linalg.norm(rep) + 1e-8),
                                        emb / (np.linalg.norm(emb) + 1e-8),
                                    )
                                )
                                if sim_rep < self.params.embedding_similarity_gate:
                                    track.visual_tracker = None
                                    return
                            if history:
                                sims = cosine_similarities(history, emb)
                                track.vt_embedding_rep = update_weighted_embedding(
                                    track.vt_embedding_rep, emb, track.score
                                )
                                if np.max(sims) < self.params.embedding_similarity_gate:
                                    track.visual_tracker = None
                                    return
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "VT update result: track=%s frame_idx=%s bbox=%s",
                            track.track_id,
                            frame_idx,
                            bbox,
                        )
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


def _cosine_similarities(history: list[np.ndarray], current: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between history embeddings and the current one."""
    if not history:
        return np.zeros(0, dtype=np.float32)
    stacked = np.stack(history, axis=0).astype(np.float32, copy=False)
    current = current.astype(np.float32, copy=False)
    norm_hist = np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-8
    norm_cur = np.linalg.norm(current) + 1e-8
    return (stacked @ current) / (norm_hist[:, 0] * norm_cur)

    def __del__(self) -> None:  # pragma: no cover - defensive
        self._reset_frame_iterator()


__all__ = ["HybridSOTTracker"]
