from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import polars as pl

from anonymizer.paths import get_models_dir
from anonymizer.sahi_integration import DEFAULT_CATEGORY_MAPPING, SahiOnnxDetectionModel

from .cancellation import CancellationException
from .io.video import get_video_info, iter_frame_batches
from .utils.progress import ProgressRateEstimator, format_progress_message

DEFAULT_MODELS_DIR = get_models_dir()


class BaseDetector:
    """Shared functionality for detectors backed by an ONNXRuntime session."""

    def __init__(
        self,
        model_path: Path | str,
        cancel_event: threading.Event | None = None,
        batch_size: int = 8,
        progress_callback: Callable[[int, str, str], None] | None = None,
        plate_threshold: float = 0.5,
        face_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        inference_size: int = 1920,
        execution_providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.cancel_event = cancel_event
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        self.plate_threshold = plate_threshold
        self.face_threshold = face_threshold
        self.logger = logging.getLogger("obscuro.detection")
        self.nms_iou_threshold = float(max(0.0, min(nms_iou_threshold, 1.0)))
        self.inference_size = max(int(inference_size), 256)

        self.requested_execution_providers, session_opts = self._get_execution_providers(
            execution_providers
        )
        self.session = self._load_model(
            self.model_path, self.requested_execution_providers, session_opts
        )
        raw_providers: list[str] = []
        if hasattr(self.session, "get_providers") and callable(self.session.get_providers):
            try:
                providers = self.session.get_providers()
            except Exception:  # pragma: no cover - defensive
                providers = None
            if isinstance(providers, list | tuple):
                raw_providers = [str(p) for p in providers]
        if not raw_providers:
            raw_providers = list(self.requested_execution_providers)
        self.active_execution_providers = raw_providers
        self.execution_provider = raw_providers[0] if raw_providers else "unknown"

        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError("Loaded ONNX model does not expose any inputs")
        self._input_name = inputs[0].name

        model_stem = Path(model_path).stem
        prefix = model_stem.split("_")[0]
        self.training_size = 640
        if prefix.isdigit():
            self.training_size = int(prefix)

        self.imgsz = int(math.ceil(self.training_size / 32) * 32)
        self.inference_size = max(self.inference_size, self.imgsz)
        self.logger.info(
            f"Model training size inferred as {self.training_size}x{self.training_size} from filename '{model_stem}'"
        )

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def detect(self, input: Any) -> pl.DataFrame:
        if isinstance(input, Path):
            return self._detect_from_path(input)
        if isinstance(input, np.ndarray):
            if input.ndim == 4:
                return self._detect_from_batch_array(input)
            if input.ndim == 3:
                return self._detect_from_array(input)
        if isinstance(input, list) and all(isinstance(x, np.ndarray) for x in input):
            return self._detect_from_list(input)
        raise ValueError("Unsupported input type for detection")

    def get_execution_provider_status(self) -> dict[str, Any]:
        primary_requested = (
            self.requested_execution_providers[0] if self.requested_execution_providers else None
        )
        primary_active = self.execution_provider if self.execution_provider else None
        status_code = (
            0 if primary_requested == primary_active and primary_active is not None else -1
        )
        return {
            "requested": list(self.requested_execution_providers),
            "active": list(self.active_execution_providers),
            "primary": primary_active,
            "status_code": status_code,
        }

    def set_thresholds(
        self,
        plate_threshold: float | None = None,
        face_threshold: float | None = None,
    ) -> None:
        if plate_threshold is not None:
            self.plate_threshold = float(max(0.0, min(plate_threshold, 1.0)))
        if face_threshold is not None:
            self.face_threshold = float(max(0.0, min(face_threshold, 1.0)))

    # --------------------------------------------------------------------- #
    # Abstract hooks                                                        #
    # --------------------------------------------------------------------- #

    def _detect_from_array(self, image: np.ndarray) -> pl.DataFrame:
        raise NotImplementedError

    def _detect_from_list(self, images: list[np.ndarray]) -> pl.DataFrame:
        raise NotImplementedError

    def _detect_from_batch_array(self, tensor: np.ndarray) -> pl.DataFrame:
        raise NotImplementedError

    def _detect_from_path(self, path: Path) -> pl.DataFrame:
        raise NotImplementedError

    # --------------------------------------------------------------------- #
    # Shared helpers                                                        #
    # --------------------------------------------------------------------- #

    def _get_execution_providers(
        self, forced_providers: list[str] | None = None
    ) -> tuple[list[str], ort.SessionOptions]:
        """
        Get execution providers for ONNX Runtime.

        :param forced_providers: If provided, use these providers directly (e.g., ["CPUExecutionProvider"]).
        :return: Tuple of (providers list, session options)
        """
        so = ort.SessionOptions()

        # If specific providers are forced, use them directly
        if forced_providers:
            self.logger.info(f"Using forced execution providers: {forced_providers}")
            return forced_providers, so

        # Otherwise, auto-detect best available provider
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            self.logger.info("Using CUDAExecutionProvider")
            providers = ["CUDAExecutionProvider"]
        elif "CoreMLExecutionProvider" in providers:
            self.logger.info("Using CoreMLExecutionProvider")
            providers = [
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "CPUAndNeuralEngine",
                        "RequireStaticInputShapes": "1",
                        "EnableOnSubgraphs": "0",
                    },
                )
            ]
        elif "MPSExecutionProvider" in providers:
            self.logger.info("Using MPSExecutionProvider")
            providers = ["MPSExecutionProvider"]
        elif "MLComputeExecutionProvider" in providers:
            self.logger.info("Using MLComputeExecutionProvider")
            providers = ["MLComputeExecutionProvider"]
        elif "DmlExecutionProvider" in providers:
            self.logger.info("Using DmlExecutionProvider")
            providers = ["DmlExecutionProvider"]
        elif "TensorrtExecutionProvider" in providers:
            self.logger.info("Using TensorrtExecutionProvider")
            providers = ["TensorrtExecutionProvider"]
        elif "OpenVINOExecutionProvider" in providers:
            self.logger.info("Using OpenVINOExecutionProvider")
            providers = ["OpenVINOExecutionProvider"]
        elif "QNNExecutionProvider" in providers:
            self.logger.info("Using QNNExecutionProvider")
            providers = ["QNNExecutionProvider"]
        else:
            self.logger.info("Using CPUExecutionProvider")
            providers = ["CPUExecutionProvider"]
        return providers, so

    def _load_model(
        self, model_path: Path | str, providers: list[str], session_opts: ort.SessionOptions
    ) -> Any:
        path = self._resolve_model_path(model_path)
        with contextlib.suppress(Exception):
            ort.preload_dlls()
        try:
            return ort.InferenceSession(str(path), providers=providers, sess_options=session_opts)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load ONNX model {path}: {exc}") from exc

    def _resolve_model_path(self, model_path: Path | str) -> Path:
        path = Path(model_path)
        candidates: Iterable[Path]
        candidates = (path,) if path.is_absolute() else (path, DEFAULT_MODELS_DIR / path)

        for candidate in candidates:
            if candidate.exists():
                resolved = candidate.resolve()
                self.logger.debug("Resolved detector model path to %s", resolved)
                self.model_path = resolved
                return resolved

        search_paths = ", ".join(str(p.resolve()) for p in candidates)
        hint = (
            "Model file not found. Checked: "
            f"{search_paths}. Upload a model via the API, place an ONNX checkpoint under "
            f"{DEFAULT_MODELS_DIR.resolve()}, or update your configuration to reference the file."
        )
        raise FileNotFoundError(hint)

    def _report_progress(self, percentage: int, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(percentage, "Detection", message)

    def _empty_result_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "frame": pl.Int64,
                "x1": pl.Float64,
                "y1": pl.Float64,
                "x2": pl.Float64,
                "y2": pl.Float64,
                "confidence": pl.Float64,
                "object_class": pl.Int64,
                "frame_width": pl.Int64,
                "frame_height": pl.Int64,
                "is_confident": pl.Boolean,
            }
        )

    def _check_cancelled(self) -> None:
        if self.cancel_event and self.cancel_event.is_set():
            raise CancellationException("Detection cancelled")

    @staticmethod
    def _ensure_channel_last(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] <= 4:
            return np.transpose(image, (1, 2, 0))
        return image

    def _build_detection_dataframe(
        self,
        *,
        frames: Iterable[int],
        x1: Iterable[float],
        y1: Iterable[float],
        x2: Iterable[float],
        y2: Iterable[float],
        scores: Iterable[float],
        classes: Iterable[int],
        frame_widths: Iterable[int],
        frame_heights: Iterable[int],
        thresholds: Iterable[float],
    ) -> pl.DataFrame:
        frame_list = list(frames)
        if not frame_list:
            return self._empty_result_df()

        score_list = list(scores)
        threshold_list = list(thresholds)
        confident = [
            float(score) >= float(thresh)
            for score, thresh in zip(score_list, threshold_list, strict=False)
        ]

        return pl.DataFrame(
            {
                "frame": frame_list,
                "x1": list(x1),
                "y1": list(y1),
                "x2": list(x2),
                "y2": list(y2),
                "confidence": score_list,
                "object_class": list(classes),
                "frame_width": list(frame_widths),
                "frame_height": list(frame_heights),
                "is_confident": confident,
            }
        )

    def _letterbox(
        self,
        image: np.ndarray,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        shape = image.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        if shape[0] == 0 or shape[1] == 0:
            raise ValueError("Cannot process empty image")

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, left = 0, 0
        bottom, right = int(dh), int(dw)
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        scale = (
            float(new_unpad[0] / max(shape[1], 1)),
            float(new_unpad[1] / max(shape[0], 1)),
        )
        pad = (float(left), float(top))

        return image, scale, pad

    def _preprocess(self, image: np.ndarray):
        original_shape = image.shape[:2]
        image, scale, pad = self._letterbox(image, new_shape=(self.imgsz, self.imgsz), auto=False)
        image = image[..., ::-1].transpose((2, 0, 1))
        image = np.ascontiguousarray(image)
        image = image.astype(np.float32) / 255.0

        if image.ndim == 3:
            image = image[None]

        meta = {
            "scale": scale,
            "pad": pad,
            "original_shape": original_shape,
        }

        return image, meta

    def _postprocess(
        self,
        outputs,
        metas: list[dict[str, Any]],
        throwaway_score: float = 0.05,
    ) -> pl.DataFrame:
        if self.cancel_event and self.cancel_event.is_set():
            raise CancellationException("Detection cancelled")

        if not metas or not outputs:
            return self._empty_result_df()

        detections = outputs[0]

        if len(detections.shape) == 3 and detections.shape[1] == 6:
            detections = detections.transpose(0, 2, 1)

        frames: list[int] = []
        x1_list: list[float] = []
        y1_list: list[float] = []
        x2_list: list[float] = []
        y2_list: list[float] = []
        confidence_list: list[float] = []
        class_list: list[int] = []
        width_list: list[int] = []
        height_list: list[int] = []
        threshold_list: list[float] = []

        for image_index in range(min(len(detections), len(metas))):
            image_detections = detections[image_index]
            if image_detections.size == 0:
                continue

            meta = metas[image_index]
            original_h, original_w = map(int, meta["original_shape"])
            scale_x, scale_y = meta["scale"]
            pad_x, pad_y = meta["pad"]
            if scale_x == 0 or scale_y == 0:
                continue

            coords = image_detections[:, :4]
            scores = image_detections[:, 4:]
            if coords.size == 0 or scores.size == 0:
                continue

            best_class = scores.argmax(axis=1)
            best_scores = scores[np.arange(scores.shape[0]), best_class]

            pre_keep_mask = best_scores >= throwaway_score
            if not np.any(pre_keep_mask):
                continue
            coords = coords[pre_keep_mask]
            best_class = best_class[pre_keep_mask]
            best_scores = best_scores[pre_keep_mask]

            thresholds = np.where(best_class == 0, self.plate_threshold, self.face_threshold)
            keep_mask = best_scores > 0.0
            if not np.any(keep_mask):
                continue

            coords = coords[keep_mask]
            best_class = best_class[keep_mask]
            best_scores = best_scores[keep_mask]
            thresholds = thresholds[keep_mask]

            half_wh = coords[:, 2:4] / 2.0
            centers = coords[:, :2]
            mins = centers - half_wh
            maxs = centers + half_wh

            mins[:, 0] = (mins[:, 0] - pad_x) / scale_x
            mins[:, 1] = (mins[:, 1] - pad_y) / scale_y
            maxs[:, 0] = (maxs[:, 0] - pad_x) / scale_x
            maxs[:, 1] = (maxs[:, 1] - pad_y) / scale_y

            np.clip(mins[:, 0], 0.0, original_w, out=mins[:, 0])
            np.clip(mins[:, 1], 0.0, original_h, out=mins[:, 1])
            np.clip(maxs[:, 0], 0.0, original_w, out=maxs[:, 0])
            np.clip(maxs[:, 1], 0.0, original_h, out=maxs[:, 1])

            widths = maxs[:, 0] - mins[:, 0]
            heights = maxs[:, 1] - mins[:, 1]
            valid_mask = (widths > 0.0) & (heights > 0.0)
            if not np.any(valid_mask):
                continue

            mins = mins[valid_mask]
            maxs = maxs[valid_mask]
            best_class = best_class[valid_mask]
            best_scores = best_scores[valid_mask]
            thresholds = thresholds[valid_mask]

            boxes_xyxy = np.column_stack((mins[:, 0], mins[:, 1], maxs[:, 0], maxs[:, 1])).astype(
                np.float32
            )
            keep_indices = _batched_nms_numpy(
                boxes_xyxy,
                best_scores.astype(np.float32),
                best_class.astype(np.int64),
                float(self.nms_iou_threshold),
            )
            if keep_indices.size == 0:
                continue

            kept_mins = mins[keep_indices]
            kept_maxs = maxs[keep_indices]
            kept_scores = best_scores[keep_indices]
            kept_classes = best_class[keep_indices]
            kept_thresh = thresholds[keep_indices]

            frames.extend([image_index] * len(keep_indices))
            x1_list.extend(kept_mins[:, 0].astype(float))
            y1_list.extend(kept_mins[:, 1].astype(float))
            x2_list.extend(kept_maxs[:, 0].astype(float))
            y2_list.extend(kept_maxs[:, 1].astype(float))
            confidence_list.extend(kept_scores.astype(float))
            class_list.extend(kept_classes.astype(int))
            width_list.extend([original_w] * len(keep_indices))
            height_list.extend([original_h] * len(keep_indices))
            threshold_list.extend(kept_thresh.astype(float))

        return self._build_detection_dataframe(
            frames=frames,
            x1=x1_list,
            y1=y1_list,
            x2=x2_list,
            y2=y2_list,
            scores=confidence_list,
            classes=class_list,
            frame_widths=width_list,
            frame_heights=height_list,
            thresholds=threshold_list,
        )


def _batched_nms_numpy(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Perform batched non-maximum suppression using NumPy primitives."""
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)

    def nms_single(indices: np.ndarray) -> list[int]:
        local_boxes = boxes[indices]
        local_scores = scores[indices]
        order = local_scores.argsort()[::-1]
        keep_local: list[int] = []

        while order.size > 0:
            current = order[0]
            keep_local.append(int(indices[current]))
            if order.size == 1:
                break
            rest = order[1:]
            ious = _compute_iou(local_boxes[current], local_boxes[rest])
            order = rest[ious <= iou_threshold]
        return keep_local

    keep: list[int] = []
    unique_classes = np.unique(classes)
    for cls in unique_classes:
        cls_indices = np.where(classes == cls)[0]
        keep.extend(nms_single(cls_indices))

    keep_sorted = sorted(keep, key=lambda idx: scores[idx], reverse=True)
    return np.asarray(keep_sorted, dtype=np.int64)


def _compute_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Compute IoU values between a single box and many boxes (XYXY format)."""
    if others.size == 0:
        return np.empty(0, dtype=np.float32)

    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_self = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_others = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
        0.0, others[:, 3] - others[:, 1]
    )
    union = area_self + area_others - inter_area
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter_area / union, 0.0)
    return iou.astype(np.float32)


class FrameDetector(BaseDetector):
    """Detector that processes full frames without tiling."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logger.info("SAHI inference disabled; using full-frame pipeline")

    def _detect_from_array(self, image: np.ndarray) -> pl.DataFrame:
        self._report_progress(0, "Processing single image")
        input_tensor, meta = self._preprocess(image)
        outputs = self.session.run(None, {self._input_name: input_tensor})
        self._report_progress(100, "Single image processing complete")
        return self._postprocess(outputs, [meta])

    def _detect_from_list(self, images: list[np.ndarray]) -> pl.DataFrame:
        self._check_cancelled()

        batch = []
        metas: list[dict[str, Any]] = []

        for image in images:
            self._check_cancelled()
            preprocessed, meta = self._preprocess(image)
            batch.append(preprocessed[0])
            metas.append(meta)
        if not batch:
            return self._empty_result_df()

        self._check_cancelled()

        input_tensor = np.stack(batch, axis=0)
        outputs = self.session.run(None, {self._input_name: input_tensor})

        return self._postprocess(outputs, metas)

    def _detect_from_batch_array(self, tensor: np.ndarray) -> pl.DataFrame:
        self._report_progress(0, "Starting batch array processing")
        batch_results: list[pl.DataFrame] = []
        total_batches = (tensor.shape[0] + self.batch_size - 1) // self.batch_size

        for i in range(0, tensor.shape[0], self.batch_size):
            try:
                self._check_cancelled()
            except CancellationException:
                self._report_progress(0, "Cancelled")
                raise

            batch = tensor[i : i + self.batch_size]
            batch_images = []
            for image in batch:
                batch_images.append(self._ensure_channel_last(image))
            df = self._detect_from_list(batch_images)
            if not df.is_empty():
                batch_results.append(df)

            percentage = int(((i // self.batch_size + 1) / total_batches) * 100)
            self._report_progress(
                percentage, f"Processed batch {i // self.batch_size + 1}/{total_batches}."
            )

        self._report_progress(100, "Batch array processing complete")
        return pl.concat(batch_results, rechunk=True) if batch_results else self._empty_result_df()

    def _detect_from_path(self, path: Path) -> pl.DataFrame:
        self.logger.debug("Starting detection for %s", path)
        self._report_progress(0, "Starting")

        results: list[pl.DataFrame] = []
        video_info = get_video_info(path)
        total_frames_meta = int(video_info.get("frame_count") or 0)
        total_batches = (
            (total_frames_meta + self.batch_size - 1) // self.batch_size
            if total_frames_meta > 0
            else None
        )

        processed_batches = 0
        processed_frames = 0
        batch_start_time = time.perf_counter()
        rate_tracker = ProgressRateEstimator()
        prefetch = max(self.batch_size * 2, 4)

        for batch in iter_frame_batches(path, self.batch_size, prefetch=prefetch):
            try:
                self._check_cancelled()
            except CancellationException:
                self._report_progress(0, "Cancelled")
                raise

            if not batch:
                continue

            frame_indices = np.array([idx for idx, _ in batch], dtype=np.int64)
            frames_bgr = [frame for _, frame in batch]

            self.logger.debug(
                "Processing batch %d (frames %s-%s)",
                processed_batches + 1,
                int(frame_indices[0]),
                int(frame_indices[-1]),
            )

            df = self._detect_from_list(frames_bgr)
            if not df.is_empty():
                local_indices = df.get_column("frame").to_numpy()
                mapped = frame_indices[local_indices]
                df = df.with_columns(pl.Series(name="frame", values=mapped, dtype=pl.Int64))
                results.append(df)

            processed_batches += 1
            processed_frames += len(batch)
            batch_duration = max(1e-6, time.perf_counter() - batch_start_time)
            fps = rate_tracker.record(len(batch), batch_duration)
            if total_frames_meta > 0:
                remaining_frames = max(total_frames_meta - processed_frames, 0)
                percentage = int(min(100, (processed_frames / total_frames_meta) * 100))
                prefix = (
                    f"Processed batch {processed_batches}/{total_batches}"
                    if total_batches
                    else f"Processed {processed_frames}/{total_frames_meta} frames"
                )
                message = format_progress_message(prefix, fps, remaining_frames)
            else:
                percentage = min(99, max(1, processed_batches))
                message = format_progress_message(
                    f"Processed {processed_frames} frames",
                    fps,
                    None,
                )
            self._report_progress(percentage, message)
            batch_start_time = time.perf_counter()

        if processed_frames == 0:
            self.logger.warning("No frames to process")
            self._report_progress(100, "No frames to process")
            return self._empty_result_df()

        self.logger.debug("Detection complete")
        self._report_progress(100, "Complete")
        df = pl.concat(results, rechunk=True) if results else self._empty_result_df()
        return df


class SahiDetector(BaseDetector):
    """Detector that performs SAHI tiled inference and merges detections."""

    def __init__(
        self,
        *args,
        sahi_overlap_ratio: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sahi_overlap_ratio = float(max(0.0, min(sahi_overlap_ratio, 0.99)))
        self._sahi_model: SahiOnnxDetectionModel | None = None
        self._category_mapping = dict(DEFAULT_CATEGORY_MAPPING)
        self.logger.info(
            "SAHI inference configured tile=%dx%d inference_size=%d overlap=%.2f",
            self.imgsz,
            self.imgsz,
            self.inference_size,
            self.sahi_overlap_ratio,
        )

    # SAHI-specific utilities -------------------------------------------------

    def _ensure_sahi_model(self) -> SahiOnnxDetectionModel:
        if self._sahi_model is None:
            self._sahi_model = SahiOnnxDetectionModel(
                detector=self,
                category_mapping=self._category_mapping,
            )
            self.logger.info("Initialized SAHI wrapper around shared ONNX session")
        return self._sahi_model

    @staticmethod
    def _prepare_image_for_sahi(image: np.ndarray) -> np.ndarray:
        prepared = image
        if prepared.ndim == 3 and prepared.shape[0] <= 4:
            prepared = np.transpose(prepared, (1, 2, 0))
        if prepared.ndim == 2 or (prepared.ndim == 3 and prepared.shape[2] == 1):
            prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
        if prepared.ndim != 3 or prepared.shape[2] != 3:
            raise ValueError("SAHI integration expects 3-channel images")
        return prepared

    def _downscale_for_sahi(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        longer_edge = max(width, height)
        if longer_edge <= self.inference_size:
            return image, 1.0
        scale = float(self.inference_size) / float(longer_edge)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _object_predictions_to_dataframe(
        self,
        predictions,
        *,
        frame_width: int,
        frame_height: int,
        original_width: int | None = None,
        original_height: int | None = None,
        scale_factor: float = 1.0,
    ) -> pl.DataFrame:
        frames: list[int] = []
        x1_list: list[float] = []
        y1_list: list[float] = []
        x2_list: list[float] = []
        y2_list: list[float] = []
        confidence_list: list[float] = []
        class_list: list[int] = []
        width_list: list[int] = []
        height_list: list[int] = []
        threshold_list: list[float] = []

        target_width = float(original_width) if original_width is not None else float(frame_width)
        target_height = (
            float(original_height) if original_height is not None else float(frame_height)
        )
        scale_multiplier = 1.0 / scale_factor if scale_factor not in (0.0, None) else 1.0

        for prediction in predictions:
            bbox = prediction.bbox
            x1, y1, x2, y2 = bbox.to_xyxy()
            shift_x = getattr(bbox, "shift_x", 0)
            shift_y = getattr(bbox, "shift_y", 0)
            x1 += shift_x
            x2 += shift_x
            y1 += shift_y
            y2 += shift_y
            if scale_multiplier != 1.0:
                x1 *= scale_multiplier
                x2 *= scale_multiplier
                y1 *= scale_multiplier
                y2 *= scale_multiplier
            x1 = float(np.clip(x1, 0.0, target_width))
            x2 = float(np.clip(x2, 0.0, target_width))
            y1 = float(np.clip(y1, 0.0, target_height))
            y2 = float(np.clip(y2, 0.0, target_height))
            if x2 <= x1 or y2 <= y1:
                continue
            score = float(prediction.score.value)
            category_id = int(prediction.category.id)
            threshold = self.plate_threshold if category_id == 0 else self.face_threshold

            frames.append(0)
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
            confidence_list.append(score)
            class_list.append(category_id)
            width_list.append(round(target_width))
            height_list.append(round(target_height))
            threshold_list.append(threshold)

        return self._build_detection_dataframe(
            frames=frames,
            x1=x1_list,
            y1=y1_list,
            x2=x2_list,
            y2=y2_list,
            scores=confidence_list,
            classes=class_list,
            frame_widths=width_list,
            frame_heights=height_list,
            thresholds=threshold_list,
        )

    def _predict_single_image(self, image: np.ndarray) -> pl.DataFrame:
        prepared = self._prepare_image_for_sahi(image)
        original_height, original_width = prepared.shape[:2]
        downscaled, scale_ratio = self._downscale_for_sahi(prepared)

        sahi_model = self._ensure_sahi_model()
        sahi_model.tile_batch_size = max(1, int(self.batch_size))
        prediction = sahi_model.predict_fused(
            image=downscaled,
            slice_height=self.imgsz,
            slice_width=self.imgsz,
            overlap_height_ratio=self.sahi_overlap_ratio,
            overlap_width_ratio=self.sahi_overlap_ratio,
            postprocess_type="NMS",
            postprocess_match_threshold=float(self.nms_iou_threshold),
            postprocess_class_agnostic=False,
            verbose=0,
        )

        if not prediction.object_prediction_list:
            return self._empty_result_df()

        return self._object_predictions_to_dataframe(
            prediction.object_prediction_list,
            frame_width=prediction.image_width,
            frame_height=prediction.image_height,
            original_width=original_width,
            original_height=original_height,
            scale_factor=scale_ratio,
        )

    def _iter_frame_sequence(
        self, images: Iterable[np.ndarray]
    ) -> Iterable[tuple[int, pl.DataFrame | None]]:
        for idx, image in enumerate(images):
            self._check_cancelled()
            df = self._predict_single_image(image)
            if df.is_empty():
                yield idx, None
            else:
                yield idx, df.with_columns(pl.lit(idx).alias("frame"))

    # Implement abstract hooks ------------------------------------------------

    def _detect_from_array(self, image: np.ndarray) -> pl.DataFrame:
        self._report_progress(0, "Processing single image")
        df = self._predict_single_image(image)
        self._report_progress(100, "Single image processing complete")
        return df

    def _detect_from_list(self, images: list[np.ndarray]) -> pl.DataFrame:
        results = [df for _, df in self._iter_frame_sequence(images) if df is not None]
        return pl.concat(results, rechunk=True) if results else self._empty_result_df()

    def _detect_from_batch_array(self, tensor: np.ndarray) -> pl.DataFrame:
        self._report_progress(0, "Starting batch array processing")
        total_frames = tensor.shape[0] if tensor.ndim > 0 else 0
        results: list[pl.DataFrame] = []
        normalized_images = (self._ensure_channel_last(image) for image in tensor)

        for idx, df in self._iter_frame_sequence(normalized_images):
            if df is not None:
                results.append(df)
            if total_frames:
                percentage = int(((idx + 1) / total_frames) * 100)
                self._report_progress(percentage, f"Processed frame {idx + 1}/{total_frames}.")

        self._report_progress(100, "Batch array processing complete")
        return pl.concat(results, rechunk=True) if results else self._empty_result_df()

    def _detect_from_path(self, path: Path) -> pl.DataFrame:
        self.logger.debug("Starting detection for %s", path)
        self._report_progress(0, "Starting")

        results: list[pl.DataFrame] = []
        video_info = get_video_info(path)
        total_frames_meta = int(video_info.get("frame_count") or 0)
        processed_frames = 0
        batch_start_time = time.perf_counter()
        rate_tracker = ProgressRateEstimator()
        prefetch = max(self.batch_size * 2, 4)

        for batch in iter_frame_batches(path, self.batch_size, prefetch=prefetch):
            try:
                self._check_cancelled()
            except CancellationException:
                self._report_progress(0, "Cancelled")
                raise
            if not batch:
                continue

            frame_indices = np.array([idx for idx, _ in batch], dtype=np.int64)
            frames_bgr = [frame for _, frame in batch]

            frame_dfs = [df for _, df in self._iter_frame_sequence(frames_bgr) if df is not None]

            if frame_dfs:
                df = pl.concat(frame_dfs, rechunk=True)
                local_indices = df.get_column("frame").to_numpy()
                mapped = frame_indices[local_indices]
                df = df.with_columns(pl.Series(name="frame", values=mapped, dtype=pl.Int64))
                results.append(df)

            processed_frames += len(batch)
            batch_duration = max(1e-6, time.perf_counter() - batch_start_time)
            fps = rate_tracker.record(len(batch), batch_duration)
            if total_frames_meta > 0:
                remaining_frames = max(total_frames_meta - processed_frames, 0)
                percentage = int(min(100, (processed_frames / total_frames_meta) * 100))
                message = format_progress_message(
                    f"Processed {processed_frames}/{total_frames_meta} frames",
                    fps,
                    remaining_frames,
                )
            else:
                percentage = min(99, max(1, processed_frames))
                message = format_progress_message(
                    f"Processed {processed_frames} frames",
                    fps,
                    None,
                )
            self._report_progress(percentage, message)
            batch_start_time = time.perf_counter()

        if processed_frames == 0:
            self.logger.warning("No frames to process")
            self._report_progress(100, "No frames to process")
            return self._empty_result_df()

        self.logger.debug("Detection complete")
        self._report_progress(100, "Complete")
        return pl.concat(results, rechunk=True) if results else self._empty_result_df()


def Detector(*args, **kwargs):
    """Factory preserving the legacy constructor."""
    use_sahi = kwargs.pop("use_sahi", False)
    overlap = kwargs.pop("sahi_overlap_ratio", 0.2)
    if use_sahi:
        kwargs["sahi_overlap_ratio"] = overlap
        return SahiDetector(*args, **kwargs)
    return FrameDetector(*args, **kwargs)


__all__ = ["Detector", "FrameDetector", "SahiDetector"]
