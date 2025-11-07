import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import polars as pl

from .cancellation import CancellationException
from .io.blur_rois import blur_rois, convert_relative_to_absolute_rois
from .io.video import blur_video_av, get_video_info
from .utils.progress import ProgressRateEstimator, format_progress_message


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

        def _rows_to_boxes(rows: Sequence[dict[str, Any]]):
            return [(float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])) for r in rows]

        def process_frame(frame: np.ndarray, frame_num: int) -> np.ndarray:
            """Process a single frame with blurring or debug overlay."""
            if self.cancel_event and self.cancel_event.is_set():
                raise CancellationException("Blurring cancelled")

            track_rows = track_rows_by_frame.get(frame_num, [])

            if debug_mode:
                detection_rows = detection_rows_by_frame.get(frame_num, [])
                self._render_debug(frame, track_rows, detection_rows)
                return frame

            if track_rows:
                boxes = _rows_to_boxes(track_rows)
                absolute_rois = self._ensure_absolute_rois(boxes, frame.shape)
                if absolute_rois:
                    frame = blur_rois(
                        frame,
                        absolute_rois,
                        blur_type=self.blur_type,
                        blur_strength=self.blur_strength,
                    )

            return frame

        def progress_update(frame_num: int, total: int, _raw_message: str):
            nonlocal last_progress_time
            if self.progress_callback:
                now = time.perf_counter()
                duration = max(1e-6, now - last_progress_time)
                last_progress_time = now
                fps = progress_rate.record(1, duration)
                percentage = int((frame_num / total) * 100) if total > 0 else min(99, frame_num)
                remaining = max(total - frame_num, 0) if total > 0 else None
                prefix = (
                    f"Processing frame {frame_num}/{total}"
                    if total > 0
                    else f"Processing frame {frame_num}"
                )
                message_with_rate = format_progress_message(prefix, fps, remaining)
                self.progress_callback(percentage, "Blurring", message_with_rate)

        blur_video_av(
            input_path=input_path,
            output_path=output_path,
            blur_func=process_frame,
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
        image = cv2.imread(str(input_path))
        processed = self._apply_detections_to_image(image, detections)
        cv2.imwrite(str(output_path), processed)

    def blur_image(self, image: np.ndarray, detections: pl.DataFrame) -> np.ndarray:
        """
        Blur an image array based on detections.

        :param image: Input image as a numpy array.
        :param detections: Polars DataFrame with columns 'x1', 'y1', 'x2', 'y2';
                           coordinates are expected to be relative (0-1).
        :return: Blurred image as a numpy array.
        """
        return self._apply_detections_to_image(image, detections)

    def blur_frame(self, frame: np.ndarray, box: tuple) -> np.ndarray:
        """
        Apply the configured blur type to a specific region of the frame.

        :param frame: Input frame as numpy array
        :param box: Bounding box as tuple (x1, y1, x2, y2) in relative coordinates (0-1)
        :return: Frame with blurred region
        """
        if self.cancel_event and self.cancel_event.is_set():
            return frame

        if isinstance(box, dict):
            if not bool(box.get("is_confident", True)):
                return frame
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
        else:
            x1, y1, x2, y2 = map(float, box)

        absolute_rois = self._ensure_absolute_rois([(x1, y1, x2, y2)], frame.shape)

        if self.blur_type == "debug":
            if absolute_rois:
                x, y, w_box, h_box = absolute_rois[0]
                self._draw_box(
                    frame,
                    {
                        "x1": float(x),
                        "y1": float(y),
                        "x2": float(x + w_box),
                        "y2": float(y + h_box),
                    },
                    (255, 0, 0),
                )
            return frame

        if absolute_rois:
            return blur_rois(
                frame, absolute_rois, blur_type=self.blur_type, blur_strength=self.blur_strength
            )

        return frame

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
        if detections:
            for det in detections:
                if not bool(det.get("is_confident", True)):
                    continue
                score = det.get("confidence")
                label = None
                if isinstance(score, int | float):
                    label = f"{score:.2f}"
                self._draw_box(frame, det, (0, 0, 255), label=label)
        if tracks:
            for track in tracks:
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

    @staticmethod
    def _ensure_absolute_rois(
        boxes: list[tuple[float, float, float, float]], frame_shape: tuple[int, ...]
    ) -> list[tuple[int, int, int, int]]:
        """Convert xyxy boxes (relative or absolute) to absolute (x, y, w, h)."""
        if not boxes:
            return []

        height, width = frame_shape[:2]
        height = max(1, int(height))
        width = max(1, int(width))

        def is_relative(box: tuple[float, float, float, float]) -> bool:
            return all(0.0 <= coord <= 1.0 for coord in box)

        if all(is_relative(box) for box in boxes):
            return convert_relative_to_absolute_rois(boxes, (height, width))

        absolute: list[tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in boxes:
            x1_px = round(x1)
            y1_px = round(y1)
            x2_px = round(x2)
            y2_px = round(y2)

            if x2_px < x1_px:
                x1_px, x2_px = x2_px, x1_px
            if y2_px < y1_px:
                y1_px, y2_px = y2_px, y1_px

            x1_px = max(0, min(x1_px, width))
            y1_px = max(0, min(y1_px, height))
            x2_px = max(0, min(x2_px, width))
            y2_px = max(0, min(y2_px, height))

            roi_w = x2_px - x1_px
            roi_h = y2_px - y1_px

            if roi_w > 0 and roi_h > 0:
                absolute.append((x1_px, y1_px, roi_w, roi_h))

        return absolute

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

        boxes = [(row["x1"], row["y1"], row["x2"], row["y2"]) for row in rows]
        absolute_rois = self._ensure_absolute_rois(boxes, image.shape)
        if absolute_rois:
            image = blur_rois(
                image,
                absolute_rois,
                blur_type=self.blur_type,
                blur_strength=self.blur_strength,
            )
        return image
