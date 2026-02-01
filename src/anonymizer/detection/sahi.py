from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl
from sahi.slicing import slice_image

from anonymizer.detection.core import BaseDetector
from anonymizer.detection.utils import ensure_channel_last, preprocess_image
from anonymizer.sahi_integration import SahiOnnxDetectionModel

from ..cancellation import CancellationException
from ..io.video import get_video_info, iter_frame_batches
from ..utils.progress import ProgressRateEstimator, format_progress_message


class SahiDetector(BaseDetector):
    """Detector that performs SAHI tiled inference and merges detections."""

    def __init__(
        self,
        *args,
        sahi_overlap_ratio: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Ensure logger is available even if BaseDetector initialisation changes order
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger("obscuro.detection.sahi")
        self.sahi_overlap_ratio = float(max(0.0, min(sahi_overlap_ratio, 0.99)))
        self._sahi_model: SahiOnnxDetectionModel | None = None
        self._category_mapping = dict(self.category_mapping)
        self._logged_tile_info = False
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
        if prepared.ndim == 2:
            prepared = np.repeat(prepared[..., None], 3, axis=2)
        elif prepared.ndim == 3 and prepared.shape[2] == 1:
            prepared = np.repeat(prepared, 3, axis=2)
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
        mask_payloads: list[dict[str, Any] | None] = []

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
            threshold = self.confidence_threshold

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
            mask_payload: dict[str, Any] | None = None
            mask_obj = getattr(prediction, "mask", None)
            if mask_obj is not None and getattr(mask_obj, "data", None) is not None:
                mask_data = np.asarray(mask_obj.data)
                if mask_data.ndim >= 2:
                    mask_resized = cv2.resize(
                        mask_data.astype(float),
                        (int(target_width), int(target_height)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    binary = (mask_resized >= 0.5).reshape(-1)
                    mask_payload = {
                        "format": "binary",
                        "size": (int(target_height), int(target_width)),
                        "data": binary,
                    }
            if mask_payload:
                mask_payloads.append(self.mask_manager.register_mask_payload(mask_payload))
            else:
                mask_payloads.append(None)

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
        return self._filter_by_categories(df)

    def _predict_single_image(self, image: np.ndarray, *, frame_id: int = 0) -> pl.DataFrame:
        prepared = self._prepare_image_for_sahi(image)
        original_height, original_width = prepared.shape[:2]
        prepared, downscale = self._downscale_for_sahi(prepared)
        scale_up = 1.0 / downscale if downscale != 0 else 1.0
        tiles = slice_image(
            image=prepared,
            slice_height=self.imgsz,
            slice_width=self.imgsz,
            overlap_height_ratio=self.sahi_overlap_ratio,
            overlap_width_ratio=self.sahi_overlap_ratio,
        )
        if not tiles:
            return self._empty_result_df()

        metas: list[dict[str, Any]] = []
        tensors: list[np.ndarray] = []
        for tile_id, tile in enumerate(tiles):
            pre, meta = preprocess_image(tile["image"], self.imgsz)
            meta["tile_offset"] = (
                float(tile["starting_pixel"][0]),
                float(tile["starting_pixel"][1]),
            )
            meta["tile_id"] = tile_id
            meta["global_shape"] = prepared.shape[:2]
            meta["scale_up"] = scale_up
            metas.append(meta)
            tensors.append(pre[0])

        detections_batches: list[np.ndarray] = []
        proto_batches: list[np.ndarray] = []
        batch_size = max(1, int(self.batch_size))
        if not self._logged_tile_info:
            batches = math.ceil(len(tensors) / batch_size) if tensors else 0
            self.logger.info(
                "SAHI tiling will run %d tiles per frame in %d batch(es) (tile=%dx%d, overlap=%.2f, batch_size=%d)",
                len(tensors),
                batches,
                self.imgsz,
                self.imgsz,
                self.sahi_overlap_ratio,
                batch_size,
            )
            self._logged_tile_info = True
        for i in range(0, len(tensors), batch_size):
            batch_tensor = np.stack(tensors[i : i + batch_size], axis=0)
            outputs = self.session.run(None, {self._input_name: batch_tensor})
            detections_batches.append(outputs[0])
            if len(outputs) > 1 and isinstance(outputs[1], np.ndarray):
                proto_batches.append(outputs[1])

        if not detections_batches:
            return self._empty_result_df()

        detections_arr = (
            np.concatenate(detections_batches, axis=0)
            if len(detections_batches) > 1
            else detections_batches[0]
        )
        outputs_final = [detections_arr]
        if proto_batches:
            proto_arr = (
                np.concatenate(proto_batches, axis=0)
                if len(proto_batches) > 1
                else proto_batches[0]
            )
            outputs_final.append(proto_arr)

        frame_ids = [int(frame_id)] * len(metas)
        df = self._postprocess(outputs_final, metas, frame_ids=frame_ids)
        if downscale != 1.0 and not df.is_empty():
            scale_up = 1.0 / downscale
            df = df.with_columns(
                pl.col("x1") * scale_up,
                pl.col("y1") * scale_up,
                pl.col("x2") * scale_up,
                pl.col("y2") * scale_up,
                pl.col("frame_width") * scale_up,
                pl.col("frame_height") * scale_up,
            )
        return df

    def _iter_frame_sequence(
        self, images: Iterable[np.ndarray], frame_ids: Iterable[int] | None = None
    ) -> Iterable[tuple[int, pl.DataFrame | None]]:
        iterable = enumerate(images) if frame_ids is None else zip(frame_ids, images, strict=False)

        for idx, image in iterable:
            self._check_cancelled()
            df = self._predict_single_image(image, frame_id=int(idx))
            if df.is_empty():
                yield idx, None
            else:
                yield idx, df

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
        normalized_images = (ensure_channel_last(image) for image in tensor)

        for idx, df in self._iter_frame_sequence(normalized_images):
            if df is not None:
                results.append(df)
            if total_frames:
                percentage = round(((idx + 1) / total_frames) * 100, 2)
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
            frames = [frame for _, frame in batch]

            frame_dfs = [
                df
                for _, df in self._iter_frame_sequence(frames, frame_ids=frame_indices.tolist())
                if df is not None
            ]

            if frame_dfs:
                df = pl.concat(frame_dfs, rechunk=True)
                local_indices = df.get_column("frame").to_numpy()
                # If frames are local indices, remap to the original frame numbers;
                # if they are already absolute frame ids, skip remapping.
                if (
                    local_indices.size > 0
                    and local_indices.min() >= 0
                    and local_indices.max() < len(frame_indices)
                ):
                    mapped = frame_indices[local_indices]
                    df = df.with_columns(pl.Series(name="frame", values=mapped, dtype=pl.Int64))
                results.append(df)

            processed_frames += len(batch)
            batch_duration = max(1e-6, time.perf_counter() - batch_start_time)
            fps = rate_tracker.record(len(batch), batch_duration)
            if total_frames_meta > 0:
                remaining_frames = max(total_frames_meta - processed_frames, 0)
                percentage = min(100.0, round((processed_frames / total_frames_meta) * 100, 2))
                message = format_progress_message(
                    f"Processed {processed_frames}/{total_frames_meta} frames",
                    fps,
                    remaining_frames,
                )
            else:
                percentage = float(min(99, max(1, processed_frames)))
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
