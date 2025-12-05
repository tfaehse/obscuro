from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from anonymizer.constants import DEFAULT_BLUR_CATEGORIES
from anonymizer.detection.masks import MaskManager
from anonymizer.detection.model import ModelLoader
from anonymizer.detection.nms import batched_nms
from anonymizer.detection.utils import (
    apply_scaling_and_padding,
    ensure_channel_last,
    preprocess_image,
)
from anonymizer.model_metadata import load_model_metadata
from anonymizer.paths import get_detection_models_dir

from ..cancellation import CancellationException
from ..io.video import get_video_info, iter_frame_batches
from ..utils.progress import ProgressRateEstimator, format_progress_message

DEFAULT_MODELS_DIR = get_detection_models_dir()


class BaseDetector:
    """Shared functionality for detectors backed by an ONNXRuntime session."""

    def __init__(
        self,
        model_path: Path | str,
        cancel_event: threading.Event | None = None,
        batch_size: int = 8,
        progress_callback: Callable[[int, str, str], None] | None = None,
        confidence_threshold: float = 0.5,
        low_score_threshold: float = 0.1,
        nms_iou_threshold: float = 0.3,
        inference_size: int = 1920,
        execution_providers: list[str] | None = None,
        categories_to_blur: Sequence[str] | None = None,
    ) -> None:
        self.cancel_event = cancel_event
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        self.confidence_threshold = float(max(0.0, min(confidence_threshold, 1.0)))
        self.low_score_threshold = float(max(0.0, min(low_score_threshold, 1.0)))
        self.logger = logging.getLogger("obscuro.detection")
        self.nms_iou_threshold = float(max(0.0, min(nms_iou_threshold, 1.0)))
        self.inference_size = max(int(inference_size), 256)

        # Initialize ModelLoader
        self.model_loader = ModelLoader(model_path, execution_providers)
        self.session = self.model_loader.session
        self.model_path = self.model_loader.model_path
        self.active_execution_providers = self.model_loader.active_execution_providers
        self.execution_provider = self.model_loader.execution_provider
        self._input_name = self.model_loader.input_name

        model_stem = Path(self.model_path).stem
        prefix = model_stem.split("_")[0]
        self.training_size = 640
        if prefix.isdigit():
            self.training_size = int(prefix)

        self.imgsz = int(math.ceil(self.training_size / 32) * 32)
        self.inference_size = max(self.inference_size, self.imgsz)
        self.logger.info(
            f"Model training size inferred as {self.training_size}x{self.training_size} from filename '{model_stem}'"
        )

        meta = load_model_metadata(self.model_path)
        self.category_mapping = {str(idx): name for idx, name in enumerate(meta["classes"])}
        self.default_blur_categories = tuple(meta.get("default_blur") or DEFAULT_BLUR_CATEGORIES)

        # Initialize MaskManager
        self.mask_manager = MaskManager(self.imgsz)

        self._class_name_by_id = {
            int(class_id): str(name)
            for class_id, name in self.category_mapping.items()
            if str(class_id).lstrip("-").isdigit()
        }
        self.num_classes = max(1, len(self._class_name_by_id))
        categories = (
            categories_to_blur if categories_to_blur is not None else self.default_blur_categories
        )
        self._allowed_class_ids = self._resolve_allowed_class_ids(categories)

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

    @staticmethod
    def _normalize_category_name(value: Any) -> str:
        return str(value).strip().lower()

    def _resolve_allowed_class_ids(self, categories: Sequence[str] | None) -> set[int] | None:
        if categories is None:
            categories = DEFAULT_BLUR_CATEGORIES
        normalized = {self._normalize_category_name(cat) for cat in categories if str(cat).strip()}
        if not normalized or "*" in normalized:
            return None

        allowed: set[int] = set()
        for class_id, name in self._class_name_by_id.items():
            if self._normalize_category_name(name) in normalized:
                allowed.add(class_id)
        for identifier in normalized:
            if identifier.isdigit():
                allowed.add(int(identifier))

        if not allowed:
            self.logger.warning(
                "Configured blur categories %s did not match known detector classes %s",
                sorted(normalized),
                sorted(self._class_name_by_id.values()),
            )
        return allowed

    def _apply_class_thresholds(
        self,
        scores: np.ndarray,
        classes: np.ndarray,
    ) -> np.ndarray:
        """
        Filter detections based on class-specific low_score_threshold.
        Currently uses a single global threshold, but structure allows for per-class expansion.
        """
        return scores >= self.low_score_threshold

    def _filter_by_categories(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty() or self._allowed_class_ids is None:
            return df
        return df.filter(pl.col("object_class").is_in(list(self._allowed_class_ids)))

    def get_execution_provider_status(self) -> dict[str, Any]:
        return self.model_loader.get_status()

    def set_thresholds(
        self,
        confidence_threshold: float | None = None,
        low_score_threshold: float | None = None,
    ) -> None:
        if confidence_threshold is not None:
            self.confidence_threshold = float(max(0.0, min(confidence_threshold, 1.0)))
        if low_score_threshold is not None:
            self.low_score_threshold = float(max(0.0, min(low_score_threshold, 1.0)))

    # --------------------------------------------------------------------- #
    # Abstract hooks                                                        #
    # --------------------------------------------------------------------- #

    def _detect_from_array(self, image: np.ndarray) -> pl.DataFrame:
        self._check_cancelled()
        image = ensure_channel_last(image)
        preprocessed, meta = preprocess_image(image, self.inference_size)

        outputs = self.session.run(None, {self._input_name: preprocessed})
        return self._postprocess(outputs, [meta])

    def _detect_from_list(
        self, images: list[np.ndarray], frame_ids: Sequence[int] | None = None
    ) -> pl.DataFrame:
        self._check_cancelled()
        if not images:
            return self._empty_result_df()

        # Preprocess all images
        preprocessed_list = []
        metas = []
        for img in images:
            img = ensure_channel_last(img)
            p_img, meta = preprocess_image(img, self.inference_size)
            preprocessed_list.append(p_img)
            metas.append(meta)

        # Stack into batch
        batch_tensor = np.concatenate(preprocessed_list, axis=0)

        outputs = self.session.run(None, {self._input_name: batch_tensor})
        return self._postprocess(outputs, metas, frame_ids)

    def _detect_from_batch_array(self, tensor: np.ndarray) -> pl.DataFrame:
        self._check_cancelled()
        # tensor is (B, H, W, C)
        if tensor.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")

        images = [tensor[i] for i in range(tensor.shape[0])]
        return self._detect_from_list(images)

    def _detect_from_path(self, path: Path) -> pl.DataFrame:
        self._check_cancelled()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Check if it's an image
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            # For image, just read and detect using RGB order
            import imageio.v3 as iio

            try:
                image = iio.imread(path)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Could not read image: {path}") from exc
            return self._detect_from_array(image)

        # Assume video
        info = get_video_info(path)
        total_frames = info.total_frames

        dfs = []
        processed_frames = 0

        # Use batch size from config or default
        batch_size = getattr(self, "batch_size", 8)

        for batch, frame_indices in iter_frame_batches(path, batch_size):
            self._check_cancelled()

            # batch is list of np.ndarray (H, W, C)
            # frame_indices is list of int

            # Detect on batch
            df = self._detect_from_list(batch, frame_ids=frame_indices)
            dfs.append(df)

            processed_frames += len(batch)
            if total_frames > 0:
                self._report_progress(
                    int(processed_frames / total_frames * 100),
                    f"Processed {processed_frames}/{total_frames} frames",
                )

        if not dfs:
            return self._empty_result_df()

        return pl.concat(dfs)

    # --------------------------------------------------------------------- #
    # Shared helpers                                                        #
    # --------------------------------------------------------------------- #

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
                "mask": pl.Int64,
            }
        )

    def _check_cancelled(self) -> None:
        if self.cancel_event and self.cancel_event.is_set():
            raise CancellationException("Detection cancelled")

    # Delegate mask methods to MaskManager
    def decode_masks_for_rows(
        self,
        rows: Sequence[dict[str, Any]],
        frame_shape: tuple[int, int] | None = None,
    ) -> list[dict[str, Any] | None]:
        return self.mask_manager.decode_masks_for_rows(rows, frame_shape)

    def clear_mask_cache(self) -> None:
        self.mask_manager.clear_mask_cache()

    def _postprocess(
        self,
        outputs,
        metas: list[dict[str, Any]],
        frame_ids: Sequence[int] | None = None,
    ) -> pl.DataFrame:
        if self.cancel_event and self.cancel_event.is_set():
            raise CancellationException("Detection cancelled")

        if not metas or not outputs:
            return self._empty_result_df()

        detections_raw = outputs[0]
        mask_proto = outputs[1]

        # Expect YOLO11-seg layout: (B, 116, 8400) -> transpose to (B, 8400, 116)
        if detections_raw.ndim != 3:
            raise ValueError(f"Unexpected detector output shape: {detections_raw.shape}")
        detections = detections_raw.transpose(0, 2, 1)

        if frame_ids is None:
            frame_ids = list(range(min(len(detections), len(metas))))

        candidates: list[dict[str, Any]] = []

        for image_index in range(min(len(detections), len(metas))):
            image_detections = detections[image_index]
            if image_detections.size == 0:
                continue

            meta = metas[image_index]
            tile_id = int(meta.get("tile_id", image_index))
            scale = meta["scale"]
            pad = meta["pad"]
            original_shape = meta["original_shape"]
            tile_offset = tuple(float(v) for v in meta.get("tile_offset", (0.0, 0.0)))

            if scale[0] == 0 or scale[1] == 0:
                continue

            coords = image_detections[:, :4]
            remainder = image_detections[:, 4:]
            if coords.size == 0 or remainder.size == 0:
                continue

            mask_dim = mask_proto.shape[1] if mask_proto is not None else 0
            if remainder.shape[1] <= mask_dim:
                raise ValueError("Detector output missing class scores before mask coefficients")

            class_count = remainder.shape[1] - mask_dim
            class_scores = remainder[:, :class_count]
            mask_coeffs = remainder[:, class_count:] if mask_dim > 0 else None

            # Limit to configured number of classes
            usable_classes = min(self.num_classes, class_scores.shape[1])
            if usable_classes <= 0:
                continue
            class_scores = class_scores[:, :usable_classes]

            # Calculate final scores (YOLO11 export already applies sigmoid in-graph)
            class_probs = class_scores

            # Store mask prototype if needed
            frame_id = int(frame_ids[image_index]) if frame_ids else image_index
            if mask_proto is not None and mask_proto.shape[0] > image_index:
                proto_slice = mask_proto[image_index]
                self.mask_manager.store_mask_proto(frame_id, tile_id, proto_slice, meta)

            # Get best class and score for each box
            best_class = class_probs.argmax(axis=1)
            best_scores = class_probs[np.arange(class_probs.shape[0]), best_class]

            # 1. Filter by low_score_threshold (class-specific logic possible here)
            keep_mask = self._apply_class_thresholds(best_scores, best_class)
            if not np.any(keep_mask):
                continue

            coords = coords[keep_mask]
            best_class = best_class[keep_mask]
            best_scores = best_scores[keep_mask]
            # The 'thresholds' array is not defined at this point, it's a constant later.
            # Removing the line `thresholds = thresholds[keep_mask]` as it would cause an error.
            if mask_coeffs is not None:
                # Assuming 'pre_keep_mask' is not defined and the intent is to filter by the current 'keep_mask'
                # If 'pre_keep_mask' was meant to be an earlier filter, it needs to be defined.
                # Sticking to the most direct interpretation given the context.
                mask_coeffs = mask_coeffs[keep_mask]

            # Convert xywh to xyxy
            half_wh = coords[:, 2:4] / 2.0
            centers = coords[:, :2]
            x1 = centers[:, 0] - half_wh[:, 0]
            y1 = centers[:, 1] - half_wh[:, 1]
            x2 = centers[:, 0] + half_wh[:, 0]
            y2 = centers[:, 1] + half_wh[:, 1]
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

            # Apply scaling and padding to get absolute coordinates
            boxes_abs = apply_scaling_and_padding(boxes_xyxy, scale, pad, original_shape)

            # Add tile offset for global coordinates
            boxes_global = boxes_abs.copy()
            boxes_global[:, [0, 2]] += tile_offset[0]
            boxes_global[:, [1, 3]] += tile_offset[1]

            # Clip global boxes to global shape
            global_h, global_w = map(
                int, meta.get("global_shape", meta.get("original_shape", original_shape))
            )
            boxes_global[:, [0, 2]] = np.clip(boxes_global[:, [0, 2]], 0.0, global_w)
            boxes_global[:, [1, 3]] = np.clip(boxes_global[:, [1, 3]], 0.0, global_h)

            # Filter invalid boxes
            widths = boxes_global[:, 2] - boxes_global[:, 0]
            heights = boxes_global[:, 3] - boxes_global[:, 1]
            valid_size = (widths > 0) & (heights > 0)

            if not np.any(valid_size):
                continue

            boxes_global = boxes_global[valid_size]
            boxes_abs = boxes_abs[valid_size]
            best_scores = best_scores[valid_size]
            best_class = best_class[valid_size]
            if mask_coeffs is not None:
                mask_coeffs = mask_coeffs[valid_size]

            # Accumulate candidates for NMS
            for j in range(len(boxes_global)):
                candidates.append(
                    {
                        "frame_id": frame_id,
                        "tile_id": tile_id,
                        "global_box": boxes_global[j],
                        "tile_box": boxes_abs[j],  # Store tile-relative absolute coords
                        "score": float(best_scores[j]),
                        "cls": int(best_class[j]),
                        "threshold": float(self.confidence_threshold),  # Pass global threshold
                        "global_shape": (global_h, global_w),
                        "meta": meta,
                        "mask_coeff": mask_coeffs[j] if mask_coeffs is not None else None,
                    }
                )

        if not candidates:
            return self._empty_result_df()

        # Prepare for NMS
        boxes_global_arr = np.stack([c["global_box"] for c in candidates], axis=0).astype(
            np.float32
        )
        scores_arr = np.asarray([c["score"] for c in candidates], dtype=np.float32)
        classes_arr = np.asarray([c["cls"] for c in candidates], dtype=np.int64)

        # Apply NMS
        keep_indices, contrib_groups = batched_nms(
            boxes_global_arr, scores_arr, classes_arr, float(self.nms_iou_threshold)
        )

        if keep_indices.size == 0:
            return self._empty_result_df()

        # Build result lists
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
        mask_payloads: list[dict[str, Any] | None] = []

        # Iterate over kept indices and their contribution groups
        # contrib_groups corresponds to keep_indices in order
        for keep_idx, group_indices in zip(keep_indices, contrib_groups, strict=False):
            cand = candidates[int(keep_idx)]

            # Final confidence check (can be done here or later, doing here for efficiency)
            # But wait, the tracker might need low-confidence detections?
            # The original code passed 'thresholds' column which was just confidence_threshold.
            # And it filtered by 'best_scores > 0.0' which is very loose.
            # We will keep all NMS survivors and let the tracker/filter handle it,
            # OR we can filter by confidence_threshold here if that's the contract.
            # The test `test_detector_filters_configured_categories` expects filtering.
            # Let's trust `_filter_by_categories` to handle class filtering,
            # and we just pass everything that survived NMS.

            frames.append(int(cand["frame_id"]))
            x1_list.append(float(cand["global_box"][0]))
            y1_list.append(float(cand["global_box"][1]))
            x2_list.append(float(cand["global_box"][2]))
            y2_list.append(float(cand["global_box"][3]))
            confidence_list.append(float(cand["score"]))
            class_list.append(int(cand["cls"]))
            width_list.append(int(cand["global_shape"][1]))
            height_list.append(int(cand["global_shape"][0]))
            threshold_list.append(float(cand["threshold"]))

            # Handle mask payload with fusion
            payload_id: int | None = None

            # Collect ingredients from all contributing boxes (including the kept one)
            ingredients: list[dict[str, Any]] = []

            # The group includes the kept index itself, usually as the first element
            for idx in group_indices:
                contributor = candidates[int(idx)]
                coeff_vec = contributor.get("mask_coeff")
                if coeff_vec is None:
                    continue

                coeff_f16 = np.asarray(coeff_vec, dtype=np.float16)
                meta_payload = contributor["meta"]

                ingredients.append(
                    {
                        "frame": int(contributor["frame_id"]),
                        "tile_id": int(contributor["tile_id"]),
                        "dtype": "float16",
                        "num_coeffs": int(coeff_f16.shape[0]),
                        "coeffs": coeff_f16.tobytes(),
                        "box": np.asarray(contributor["tile_box"]).astype(float).tolist(),
                        "meta": {
                            "pad": tuple(meta_payload.get("pad", (0.0, 0.0))),
                            "scale": tuple(meta_payload.get("scale", (1.0, 1.0))),
                            "original_shape": tuple(
                                meta_payload.get(
                                    "global_shape",
                                    meta_payload.get("original_shape", (0, 0)),
                                )
                            ),
                            "offset": tuple(meta_payload.get("tile_offset", (0.0, 0.0))),
                            "imgsz": self.imgsz,
                        },
                    }
                )

            if ingredients:
                payload = {
                    "format": "coeff_ingredients",
                    "ingredients": ingredients,
                    # Use the kept box as the primary box for the payload
                    "box": np.asarray(cand["tile_box"]).astype(float).tolist(),
                }
                payload_id = self.mask_manager.register_mask_payload(payload)

            mask_payloads.append(payload_id)

        df = self._build_detection_dataframe(
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
            masks=mask_payloads,
        )

        # Finally filter by configured categories
        return self._filter_by_categories(df)

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
        masks: Iterable[dict[str, Any] | None] | None = None,
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

        if masks is None:
            mask_values: list[int | None] = [None] * len(frame_list)
        else:
            mask_values = list(masks)
            if len(mask_values) != len(frame_list):
                raise ValueError("Length of masks must match number of detections")

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
                "mask": mask_values,
            }
        )


class FrameDetector(BaseDetector):
    """Detector that processes full frames without tiling."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logger.info("SAHI inference disabled; using full-frame pipeline")

    def _detect_from_array(self, image: np.ndarray) -> pl.DataFrame:
        self._report_progress(0, "Processing single image")
        input_tensor, meta = preprocess_image(image, self.imgsz)
        outputs = self.session.run(None, {self._input_name: input_tensor})
        self._report_progress(100, "Single image processing complete")
        return self._postprocess(outputs, [meta], frame_ids=[0])

    def _detect_from_list(
        self, images: list[np.ndarray], frame_ids: Sequence[int] | None = None
    ) -> pl.DataFrame:
        self._check_cancelled()

        batch = []
        metas: list[dict[str, Any]] = []

        for image in images:
            self._check_cancelled()
            preprocessed, meta = preprocess_image(image, self.imgsz)
            batch.append(preprocessed[0])
            metas.append(meta)
        if not batch:
            return self._empty_result_df()

        self._check_cancelled()

        input_tensor = np.stack(batch, axis=0)
        outputs = self.session.run(None, {self._input_name: input_tensor})

        return self._postprocess(outputs, metas, frame_ids=frame_ids)

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
                batch_images.append(ensure_channel_last(image))
            frame_ids = list(range(i, i + len(batch_images)))
            df = self._detect_from_list(batch_images, frame_ids=frame_ids)
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

            df = self._detect_from_list(frames_bgr, frame_ids=frame_indices.tolist())
            if not df.is_empty():
                local_indices = df.get_column("frame").to_numpy()
                if local_indices.size and local_indices.max() < len(frame_indices):
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
