import contextlib
import logging
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import cv2
import imageio.v3 as iio
import numpy as np
import polars as pl

from .cancellation import CancellationException
from .io.blur_rois import blur_rois, convert_relative_to_absolute_rois
from .io.video import blur_video_av, get_video_info
from .utils.progress import ProgressRateEstimator, format_progress_message

logger = logging.getLogger("obscuro.blurring")


class MaskRegion(TypedDict):
    mask: np.ndarray
    x1: int
    y1: int


class Blurrer:
    # Available blur types
    AVAILABLE_BLUR_TYPES: ClassVar[list[str]] = [
        "gaussian",
        "pixelate",
        "blackout",
        "debug",
    ]

    @classmethod
    def get_available_blur_types(cls) -> list[str]:
        """
        Get list of available blur types.

        :return: List of available blur type strings
        """
        return cls.AVAILABLE_BLUR_TYPES.copy()

    @classmethod
    def is_valid_blur_type(cls, blur_type: str) -> bool:
        """
        Check if a blur type is valid.

        :param blur_type: Blur type to validate
        :return: True if valid, False otherwise
        """
        return blur_type.lower() in cls.AVAILABLE_BLUR_TYPES

    def __init__(
        self,
        blur_type: str = "gaussian",
        blur_strength: int = 10,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[int, str, str], None] | None = None,
    ):
        """
        Initialize the Blurrer with the specified blur type and strength.

        :param blur_type: Type of blur to apply. Options:
                         - "gaussian": Gaussian blur (smooth)
                         - "pixelate": Pixelation effect (blocky)
                         - "blackout": Solid black rectangle
                         - "debug": Draw detection (red) and track (blue) boxes with IDs
        :param blur_strength: Intensity of the blur:
                             - For gaussian: kernel size (will be made odd)
                             - For pixelate: pixel block size (higher = more pixelated)
                             - For blackout: not used
        :param cancel_event: Optional threading.Event to allow cancellation.
        :param progress_callback: Optional callback to update progress (percentage, message).
        """
        self.blur_type = blur_type.lower()
        if not self.is_valid_blur_type(self.blur_type):
            raise ValueError(
                f"Invalid blur type '{blur_type}'. Available types: {self.get_available_blur_types()}"
            )

        self.blur_strength = max(1, blur_strength)  # Ensure positive blur strength
        self.cancel_event = cancel_event
        self.progress_callback = progress_callback
        self.mask_decoder: Any | None = None

    def set_mask_decoder(self, decoder: Any | None) -> None:
        self.mask_decoder = decoder

    def set_blur_settings(self, blur_type: str | None = None, blur_strength: int | None = None):
        """
        Update blur settings at runtime.

        :param blur_type: New blur type (if provided)
        :param blur_strength: New blur strength (if provided)
        """
        if blur_type is not None:
            blur_type = blur_type.lower()
            if not self.is_valid_blur_type(blur_type):
                raise ValueError(
                    f"Invalid blur type '{blur_type}'. Available types: {self.get_available_blur_types()}"
                )
            self.blur_type = blur_type

        if blur_strength is not None:
            self.blur_strength = max(1, blur_strength)

    def blur_video(
        self,
        input_path: Path,
        tracks: pl.DataFrame,
        output_path: Path,
        *,
        codec: str | None = None,
        quality: int | None = None,
        raw_detections: pl.DataFrame | None = None,
    ) -> None:
        """
        Process the video at input_path, applying the selected blur mode (or debug overlay) to
        the regions specified by the tracker output.

        :param input_path: Path to the input video.
        :param tracks: Polars DataFrame describing tracked regions (one row per frame/track).
        :param output_path: Path for the output video.
        :param raw_detections: Optional Polars DataFrame containing detector outputs for debug overlays.
        """

        # Fetch metadata (frame count etc.) and keep for progress reporting
        video_info = get_video_info(input_path)
        video_info["frame_count"]

        track_rows_by_frame = self._group_rows_by_frame(tracks)
        detection_rows_by_frame = (
            self._group_rows_by_frame(raw_detections) if raw_detections is not None else {}
        )

        debug_mode = self.blur_type == "debug"
        progress_rate = ProgressRateEstimator()
        last_progress_time = time.perf_counter()

        def process_frame(frame: np.ndarray, frame_num: int) -> np.ndarray:
            """Process a single frame with blurring or debug overlay."""
            if self.cancel_event and self.cancel_event.is_set():
                raise CancellationException("Blurring cancelled")

            frame_start = time.perf_counter()
            mask_count = 0
            box_count = 0
            track_rows = track_rows_by_frame.get(frame_num, [])

            if debug_mode:
                detection_rows = detection_rows_by_frame.get(frame_num, [])
                self._render_debug(frame, track_rows, detection_rows)
                duration_ms = (time.perf_counter() - frame_start) * 1000.0
                logger.debug("Processed frame %d in %.2f ms (debug)", frame_num, duration_ms)
                return frame

            if track_rows:
                mask_list, rows_without_mask = self._split_mask_rows(
                    track_rows, frame.shape, frame_num
                )
                mask_count = len(mask_list)
                box_count = len(rows_without_mask)
                final_mask = self._build_frame_mask(frame.shape, mask_list, rows_without_mask)
                if np.any(final_mask):
                    blurred_full = self._apply_blur_to_roi(frame.copy())
                    frame[final_mask] = blurred_full[final_mask]
            duration_ms = (time.perf_counter() - frame_start) * 1000.0
            duration_ms = (time.perf_counter() - frame_start) * 1000.0
            logger.debug(
                "Processed frame %d in %.2f ms (mask_rows=%d, box_rows=%d)",
                frame_num,
                duration_ms,
                mask_count,
                box_count,
            )
            return frame

        def progress_update(frame_num: int, total: int, _raw_message: str):
            nonlocal last_progress_time
            if self.progress_callback:
                now = time.perf_counter()
                duration = max(1e-6, now - last_progress_time)
                last_progress_time = now
                fps = progress_rate.record(1, duration)
                if total > 0:
                    percentage = min(100.0, round((frame_num / total) * 100, 2))
                else:
                    percentage = float(min(99, frame_num))
                remaining = max(total - frame_num, 0) if total > 0 else None
                prefix = (
                    f"Processing frame {frame_num}/{total}"
                    if total > 0
                    else f"Processing frame {frame_num}"
                )
                message_with_rate = format_progress_message(prefix, fps, remaining)
                self.progress_callback(percentage, "Blurring", message_with_rate)

        def blur_frame(frame: np.ndarray, frame_num: int) -> np.ndarray:
            processed = process_frame(frame, frame_num)
            if self.mask_decoder and hasattr(self.mask_decoder, "release_mask_proto"):
                with contextlib.suppress(Exception):
                    self.mask_decoder.release_mask_proto(frame_num)
            return processed

        blur_video_av(
            input_path=input_path,
            output_path=output_path,
            blur_func=blur_frame,
            codec=codec or "h264",
            quality=quality,
            progress_callback=progress_update,
        )

        if self.progress_callback:
            self.progress_callback(100, "Blurring", "Complete")

    def blur_image_file(
        self, input_path: Path, detections: pl.DataFrame, output_path: Path
    ) -> None:
        """
        Blur an image file based on detections.

        :param input_path: Path to the input image.
        :param detections: Polars DataFrame with columns 'x1', 'y1', 'x2', 'y2';
                           coordinates are expected to be relative (0-1).
        :param output_path: Path for the output image.
        """
        image = iio.imread(input_path)
        processed = self._apply_detections_to_image(image, detections)
        iio.imwrite(output_path, processed)

    def blur_image(self, image: np.ndarray, detections: pl.DataFrame) -> np.ndarray:
        """
        Blur an image array based on detections.

        :param image: Input image as a numpy array.
        :param detections: Polars DataFrame with columns 'x1', 'y1', 'x2', 'y2';
                           coordinates are expected to be relative (0-1).
        :return: Blurred image as a numpy array.
        """
        return self._apply_detections_to_image(image, detections)

    @staticmethod
    def _group_rows_by_frame(df: pl.DataFrame | None) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        if df is None or df.is_empty():
            return grouped
        data = df.sort("frame") if "frame" in df.columns else df
        for row in data.iter_rows(named=True):
            frame_idx = int(row.get("frame", 0))
            grouped.setdefault(frame_idx, []).append(row)
        return grouped

    def _render_debug(
        self,
        frame: np.ndarray,
        tracks: Sequence[dict[str, Any]] | None,
        detections: Sequence[dict[str, Any]] | None,
    ) -> None:
        # Draw detections (red) with masks if available
        if detections:
            det_masks: list[MaskRegion | None] = []
            if self.mask_decoder:
                with contextlib.suppress(Exception):
                    det_masks = self.mask_decoder.decode_masks_for_rows(detections, frame.shape[:2])
            for idx, det in enumerate(detections):
                if not bool(det.get("is_confident", True)):
                    continue
                if det_masks and idx < len(det_masks):
                    self._draw_mask(frame, det_masks[idx], (0, 0, 255))
                else:
                    mask_region = self._decode_mask_payload(
                        det.get("mask"), frame.shape, det, frame_num=int(det.get("frame", 0))
                    )
                    self._draw_mask(frame, mask_region, (0, 0, 255))
                score = det.get("confidence")
                label = None
                if isinstance(score, int | float):
                    label = f"{score:.2f}"
                self._draw_box(frame, det, (0, 0, 255), label=label)
        # Draw tracks (blue) with masks if available
        if tracks:
            track_masks: list[MaskRegion | None] = []
            if self.mask_decoder:
                with contextlib.suppress(Exception):
                    track_masks = self.mask_decoder.decode_masks_for_rows(tracks, frame.shape[:2])
            for idx, track in enumerate(tracks):
                if track_masks and idx < len(track_masks):
                    self._draw_mask(frame, track_masks[idx], (255, 0, 0))
                else:
                    mask_region = self._decode_mask_payload(
                        track.get("mask"), frame.shape, track, frame_num=int(track.get("frame", 0))
                    )
                    self._draw_mask(frame, mask_region, (255, 0, 0))
                label_value = track.get("track_id")
                label = str(label_value) if label_value is not None else None
                raw_color = track.get("debug_color")
                if raw_color is not None:
                    if isinstance(raw_color, list | tuple) and len(raw_color) == 3:
                        color = tuple(int(c) for c in raw_color)
                    else:
                        color = (255, 0, 0)
                else:
                    color = (255, 0, 0)
                self._draw_box(frame, track, color, label=label)

    def _draw_box(
        self,
        frame: np.ndarray,
        row: dict[str, Any],
        color: tuple[int, int, int],
        label: str | None = None,
    ) -> None:
        if frame.size == 0:
            return
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return

        x1 = float(row.get("x1", 0.0))
        y1 = float(row.get("y1", 0.0))
        if "x2" in row and "y2" in row:
            x2 = float(row.get("x2", 0.0))
            y2 = float(row.get("y2", 0.0))
        else:
            width_val = float(row.get("width", 0.0))
            height_val = float(row.get("height", 0.0))
            x2 = x1 + width_val
            y2 = y1 + height_val

        if max(x1, y1, x2, y2) <= 1.0:
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height

        if x2 <= x1 or y2 <= y1:
            return

        x1_i = int(np.clip(round(x1), 0, width - 1))
        y1_i = int(np.clip(round(y1), 0, height - 1))
        x2_i = int(np.clip(round(x2), 0, width - 1))
        y2_i = int(np.clip(round(y2), 0, height - 1))
        if x2_i <= x1_i or y2_i <= y1_i:
            return

        thickness = max(1, min(width, height) // 400 + 1)
        cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, thickness, lineType=cv2.LINE_8)

        if label:
            text_origin = (x1_i, max(y1_i - 5, 0))
            cv2.putText(
                frame,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness=1,
                lineType=cv2.LINE_8,
            )

    def _draw_mask(
        self, frame: np.ndarray, mask_region: MaskRegion | None, color: tuple[int, int, int]
    ) -> None:
        region = self._normalize_mask_region(mask_region)
        if region is None:
            return
        mask = np.asarray(region["mask"], dtype=np.uint8)
        if mask.ndim != 2 or mask.size == 0:
            return
        y1 = int(region["y1"])
        x1 = int(region["x1"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        for contour in contours:
            contour[:, 0, 0] += x1
            contour[:, 0, 1] += y1
        cv2.drawContours(frame, contours, -1, color, thickness=2, lineType=cv2.LINE_8)

    def _split_mask_rows(
        self,
        rows: Sequence[dict[str, Any]],
        frame_shape: tuple[int, ...],
        frame_num: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Categorize rows into those with mask data and those without.

        Returns row dictionaries with mask payloads. Actual mask decoding
        is deferred to _build_frame_mask for lazy evaluation.
        """
        with_mask_data: list[dict[str, Any]] = []
        without_mask: list[dict[str, Any]] = []

        for row in rows:
            payload = row.get("mask")
            # Check if there's mask data to decode (either mask ID or dict payload)
            has_mask_data = payload is not None and isinstance(payload, int | dict)

            if has_mask_data:
                with_mask_data.append(row)
            else:
                without_mask.append(row)

        return with_mask_data, without_mask

    def _decode_mask_payload(
        self,
        payload: Any,
        frame_shape: tuple[int, ...],
        row: dict[str, Any] | None = None,
        frame_num: int = 0,
    ) -> MaskRegion | None:
        if payload is None:
            return None
        height, width = frame_shape[:2]
        if height <= 0 or width <= 0:
            return None
        if isinstance(payload, int) and self.mask_decoder:
            with contextlib.suppress(Exception):
                decoded = self.mask_decoder.decode_mask_from_payload(
                    payload, row or {}, frame_shape[:2]
                )
                return self._normalize_mask_region(decoded)
        if isinstance(payload, dict):
            fmt = str(payload.get("format", "")).lower()
            if fmt == "relative_polygon":
                return self._mask_region_from_polygon(
                    payload.get("points"), frame_shape, relative=True
                )
            if fmt == "absolute_polygon":
                return self._mask_region_from_polygon(
                    payload.get("points"), frame_shape, relative=False
                )
            if fmt == "binary":
                return self._mask_region_from_binary(payload, frame_shape)
            if fmt == "proto_mask":
                return self._mask_region_from_proto(payload, frame_shape, row)
        return None

    @staticmethod
    def _normalize_mask_region(value: Any) -> MaskRegion | None:
        if not isinstance(value, dict):
            return None
        mask = value.get("mask")
        x1 = value.get("x1")
        y1 = value.get("y1")
        if mask is None or x1 is None or y1 is None:
            return None
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.ndim != 2 or mask_arr.size == 0 or not np.any(mask_arr):
            return None
        try:
            x1_i = int(x1)
            y1_i = int(y1)
        except (ValueError, TypeError):
            return None
        return {"mask": mask_arr, "x1": x1_i, "y1": y1_i}

    def _mask_region_from_polygon(
        self,
        points: Any,
        frame_shape: tuple[int, ...],
        *,
        relative: bool,
    ) -> MaskRegion | None:
        if points is None:
            return None
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return None
        height, width = frame_shape[:2]
        coords = pts.copy()
        if relative:
            coords[:, 0] *= width
            coords[:, 1] *= height
        coords[:, 0] = np.clip(coords[:, 0], 0, width)
        coords[:, 1] = np.clip(coords[:, 1], 0, height)
        x_min = int(np.floor(coords[:, 0].min()))
        x_max = int(np.ceil(coords[:, 0].max()))
        y_min = int(np.floor(coords[:, 1].min()))
        y_max = int(np.ceil(coords[:, 1].max()))
        roi_w = max(0, x_max - x_min)
        roi_h = max(0, y_max - y_min)
        if roi_w == 0 or roi_h == 0:
            return None
        shifted = coords.copy()
        shifted[:, 0] -= x_min
        shifted[:, 1] -= y_min
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted.astype(np.int32)], (1,))
        if not np.any(mask):
            return None
        return {"mask": mask.astype(bool), "x1": x_min, "y1": y_min}

    def _mask_region_from_binary(
        self, payload: dict[str, Any], frame_shape: tuple[int, ...]
    ) -> MaskRegion | None:
        data = payload.get("data")
        size = payload.get("size")
        if data is None or size is None:
            return None
        arr = np.asarray(data, dtype=bool)
        try:
            shape = tuple(int(v) for v in size)
        except (ValueError, TypeError):
            return None
        try:
            mask = arr.reshape(shape)
        except ValueError:
            return None
        if mask.ndim != 2:
            return None
        return self._shrink_full_mask(mask, frame_shape)

    def _mask_region_from_proto(
        self,
        payload: dict[str, Any],
        frame_shape: tuple[int, ...],
        row: dict[str, Any] | None,
    ) -> MaskRegion | None:
        data = payload.get("data")
        size = payload.get("size")
        if not isinstance(data, bytes | bytearray) or not size:
            return None
        try:
            ph, pw = map(int, size)
        except (ValueError, TypeError):
            return None
        expected = ph * pw
        arr = np.frombuffer(data, dtype=np.uint8, count=expected)
        if arr.size != expected:
            return None
        proto_mask = arr.reshape(ph, pw).astype(np.float32) / 255.0
        if row:
            width = frame_shape[1]
            height = frame_shape[0]
            x1_raw = float(row.get("x1", 0.0))
            y1_raw = float(row.get("y1", 0.0))
            x2_raw = float(row.get("x2", x1_raw))
            y2_raw = float(row.get("y2", y1_raw))
            if max(x1_raw, y1_raw, x2_raw, y2_raw) <= 1.0:
                x1_raw *= width
                x2_raw *= width
                y1_raw *= height
                y2_raw *= height
            x1 = int(np.clip(np.floor(x1_raw), 0, width))
            y1 = int(np.clip(np.floor(y1_raw), 0, height))
            x2 = int(np.clip(np.ceil(x2_raw), 0, width))
            y2 = int(np.clip(np.ceil(y2_raw), 0, height))
            roi_w = max(0, x2 - x1)
            roi_h = max(0, y2 - y1)
            if roi_w == 0 or roi_h == 0:
                return None
            resized = cv2.resize(proto_mask, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            binary = resized >= 0.5
            if not np.any(binary):
                return None
            return {"mask": binary.astype(bool), "x1": x1, "y1": y1}
        width = frame_shape[1]
        height = frame_shape[0]
        resized = cv2.resize(proto_mask, (width, height), interpolation=cv2.INTER_LINEAR)
        binary = resized >= 0.5
        return self._shrink_full_mask(binary, frame_shape)

    def _mask_region_from_bbox(
        self, mask: np.ndarray, frame_shape: tuple[int, ...], row: dict[str, Any]
    ) -> MaskRegion | None:
        x1 = float(row.get("x1", 0.0))
        y1 = float(row.get("y1", 0.0))
        x2 = float(row.get("x2", x1))
        y2 = float(row.get("y2", y1))
        return self._extract_roi_from_mask(
            mask,
            frame_shape,
            x1=int(np.floor(x1)),
            y1=int(np.floor(y1)),
            x2=int(np.ceil(x2)),
            y2=int(np.ceil(y2)),
        )

    def _shrink_full_mask(
        self, mask: np.ndarray, frame_shape: tuple[int, ...]
    ) -> MaskRegion | None:
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim != 2 or not np.any(mask_bool):
            return None
        ys, xs = np.where(mask_bool)
        x1 = int(xs.min())
        x2 = int(xs.max() + 1)
        y1 = int(ys.min())
        y2 = int(ys.max() + 1)
        return self._extract_roi_from_mask(mask_bool, frame_shape, x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def _extract_roi_from_mask(
        mask: np.ndarray,
        frame_shape: tuple[int, ...],
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> MaskRegion | None:
        height, width = frame_shape[:2]
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0 or not np.any(roi):
            return None
        return {"mask": roi.astype(bool), "x1": x1, "y1": y1}

    def _apply_masks_to_frame(self, frame: np.ndarray, masks: Sequence[MaskRegion]) -> np.ndarray:
        final_mask = self._build_frame_mask(frame.shape, masks, [])
        if np.any(final_mask):
            blurred_full = self._apply_blur_to_roi(frame.copy())
            frame[final_mask] = blurred_full[final_mask]
        return frame

    def _build_frame_mask(
        self,
        frame_shape: tuple[int, ...],
        mask_rows: Sequence[dict[str, Any]],
        rows_without_mask: Sequence[dict[str, Any]],
    ) -> np.ndarray:
        """
        Build a combined mask for blurring.

        Decodes masks on-demand from row dictionaries with mask payloads.
        Uses batching for efficiency when decoding multiple masks.
        """
        height, width = frame_shape[:2]
        full_mask = np.zeros((height, width), dtype=bool)

        # Decode masks on-demand with batching
        if mask_rows and self.mask_decoder:
            decode_start = time.perf_counter()
            try:
                decoded_masks = self.mask_decoder.decode_masks_for_rows(mask_rows, (height, width))
            except Exception as e:
                logger.warning("Failed to decode masks: %s", e)
                decoded_masks = []
            finally:
                duration_ms = (time.perf_counter() - decode_start) * 1000.0
                logger.debug(
                    "Decoded %d masks in %.2f ms (batched)",
                    len(mask_rows),
                    duration_ms,
                )

            # Apply decoded masks to the frame mask
            for region in decoded_masks:
                if region is None:
                    continue
                normalized = self._normalize_mask_region(region)
                if normalized is None:
                    continue
                mask = np.asarray(normalized["mask"], dtype=bool)
                if mask.ndim != 2 or not np.any(mask):
                    continue
                y1 = int(np.clip(normalized["y1"], 0, height))
                x1 = int(np.clip(normalized["x1"], 0, width))
                y2 = min(height, y1 + mask.shape[0])
                x2 = min(width, x1 + mask.shape[1])
                if x2 <= x1 or y2 <= y1:
                    continue
                full_mask[y1:y2, x1:x2] |= mask[: y2 - y1, : x2 - x1]

        # Handle bounding boxes without masks (fall back to box-based blur)
        if rows_without_mask:
            boxes = [
                (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
                for r in rows_without_mask
            ]
            absolute_rois = self._ensure_absolute_rois(boxes, frame_shape)
            for x1, y1, x2, y2 in absolute_rois:
                full_mask[y1:y2, x1:x2] = True

        return full_mask

    def _blur_mask_region(self, frame: np.ndarray, mask_region: MaskRegion) -> np.ndarray:
        region = self._normalize_mask_region(mask_region)
        if region is None:
            return frame
        mask = region["mask"]
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if not np.any(mask):
            return frame
        height, width = frame.shape[:2]
        y1 = int(np.clip(region["y1"], 0, height))
        x1 = int(np.clip(region["x1"], 0, width))
        y2 = min(height, y1 + mask.shape[0])
        x2 = min(width, x1 + mask.shape[1])
        if x2 <= x1 or y2 <= y1:
            return frame
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame
        mask_slice = mask[: y2 - y1, : x2 - x1]
        if not np.any(mask_slice):
            return frame
        blurred = self._apply_blur_to_roi(roi.copy())
        roi[mask_slice] = blurred[mask_slice]
        frame[y1:y2, x1:x2] = roi
        return frame

    def _apply_blur_to_roi(self, roi: np.ndarray) -> np.ndarray:
        if roi.size == 0:
            return roi
        h, w = roi.shape[:2]
        rois = [(0, 0, w, h)]
        return blur_rois(
            roi,
            rois,
            blur_type=self.blur_type,
            blur_strength=self.blur_strength,
        )

    @staticmethod
    def _ensure_absolute_rois(
        boxes: list[tuple[float, float, float, float]], frame_shape: tuple[int, ...]
    ) -> list[tuple[int, int, int, int]]:
        """Convert normalized xyxy boxes to absolute (x, y, w, h)."""
        if not boxes:
            return []

        height, width = frame_shape[:2]
        height = max(1, int(height))
        width = max(1, int(width))
        normalized = []
        for x1, y1, x2, y2 in boxes:
            x1_f = float(x1)
            y1_f = float(y1)
            x2_f = float(x2)
            y2_f = float(y2)
            if max(x1_f, y1_f, x2_f, y2_f) > 1.0:
                x1_f /= width
                y1_f /= height
                x2_f /= width
                y2_f /= height
            nx1 = min(x1_f, x2_f)
            ny1 = min(y1_f, y2_f)
            nx2 = max(x1_f, x2_f)
            ny2 = max(y1_f, y2_f)
            normalized.append((nx1, ny1, nx2, ny2))
        return convert_relative_to_absolute_rois(normalized, (height, width))

    def _apply_detections_to_image(
        self, image: np.ndarray | None, detections: pl.DataFrame
    ) -> np.ndarray:
        """Apply blur/debug overlays to an image array based on detections."""
        if image is None:
            raise ValueError("Failed to load image for blurring")

        if "is_confident" not in detections.columns:
            detections = detections.with_columns(pl.lit(True).alias("is_confident"))

        confident = detections.filter(pl.col("is_confident"))
        rows = list(confident.iter_rows(named=True))

        if self.blur_type == "debug":
            self._render_debug(image, rows, [])
            return image

        mask_list, rows_without_mask = self._split_mask_rows(rows, image.shape, frame_num=0)
        final_mask = self._build_frame_mask(image.shape, mask_list, rows_without_mask)
        if np.any(final_mask):
            blurred_full = self._apply_blur_to_roi(image.copy())
            image[final_mask] = blurred_full[final_mask]

        if self.mask_decoder and hasattr(self.mask_decoder, "release_mask_proto"):
            frame_ids = {int(row.get("frame", 0)) for row in rows}
            for fid in frame_ids:
                with contextlib.suppress(Exception):
                    self.mask_decoder.release_mask_proto(fid)

        return image
