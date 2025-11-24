from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .blurring import Blurrer
from .cancellation import CancellationException, CancellationMixin
from .config import (
    AnonymizerConfig,
    BlurType,
    TrackerParams,
    TrackerType,
    enforce_model_batch_constraints,
    get_config,
)
from .detection import Detector
from .tracking import TrackerFactory, link_tracklets
from .tracking.common import normalized_center
from .utils.progress import throttle_progress_callback

# Type alias for numpy arrays used in the public API
NDArrayUint8 = np.ndarray[Any, np.dtype[np.uint8]]

OFFLINE_LINK_DEBUG_COLOR = (0, 255, 255)
logger = logging.getLogger(__name__)


class Anonymizer(CancellationMixin):
    def __init__(
        self,
        config: AnonymizerConfig | None = None,
        progress_callback: Callable[[int, str, str], None] | None = None,
        cancel_event: threading.Event | None = None,
        execution_providers: list[str] | None = None,
    ):
        """
        Initialize the Anonymizer with configuration and optional hooks.

        :param config: Configuration object. If None, uses global config.
        :param progress_callback: A callback function with signature (percentage: int, stage: str, message: str).
        :param cancel_event: A threading.Event instance used to signal cancellation.
        :param execution_providers: Optional list of ONNX Runtime execution providers to force (e.g., ["CPUExecutionProvider"]).
        """
        super().__init__(cancel_event=cancel_event, progress_callback=None)

        self._progress_interval_seconds = 2.0
        self._progress_percent_step = 5.0
        self._raw_progress_callback = progress_callback
        self.progress_callback = throttle_progress_callback(
            progress_callback,
            interval_seconds=self._progress_interval_seconds,
            percent_step=self._progress_percent_step,
        )

        if config is None:
            config = get_config()

        self.config = config
        enforce_model_batch_constraints(self.config, log=logger)
        self._apply_logging_preferences()

        # Initialize components using configuration
        try:
            model_path = self.config.model.path
        except ValueError as exc:
            raise FileNotFoundError(
                "No model configured. Upload a model via the API or configure model.name/model.file."
            ) from exc

        self.detector = Detector(
            model_path,
            cancel_event=self.cancel_event,
            progress_callback=self.progress_callback,
            confidence_threshold=self.config.detection.confidence_threshold,
            low_score_threshold=self.config.detection.low_score_threshold,
            batch_size=self.config.detection.batch_size,
            use_sahi=self.config.detection.use_sahi,
            inference_size=self.config.detection.inference_size,
            sahi_overlap_ratio=self.config.detection.sahi_overlap_ratio,
            execution_providers=execution_providers,
            categories_to_blur=self.config.detection.classes_to_blur,
        )

        # Create tracker using factory (video_source will be set later)
        tracker_kwargs = self.config.get_tracker_kwargs()
        self.tracker = TrackerFactory.get(
            name=self.config.tracking.type,
            video_source=None,  # Will be set before tracking
            cancel_event=self.cancel_event,
            progress_callback=self.progress_callback,
            confidence_threshold=self.config.detection.confidence_threshold,
            low_score_threshold=self.config.detection.low_score_threshold,
            **tracker_kwargs,
        )
        self.tracker.progress_callback = self.progress_callback

        self._offline_linker_tracks = None

        self.blurrer = Blurrer(
            self.config.blur.type.value,
            self.config.blur.strength,
            cancel_event=self.cancel_event,
            progress_callback=self.progress_callback,
        )
        self.blurrer.set_mask_decoder(self.detector)

        # Ensure component hooks stay in sync with runtime-provided callbacks
        self.set_runtime_hooks(cancel_event=cancel_event, progress_callback=progress_callback)

    @classmethod
    def get_available_blur_types(cls) -> list[str]:
        """Get list of available blur types."""
        return Blurrer.get_available_blur_types()

    def update_blur_settings(
        self, blur_type: str | None = None, blur_strength: int | None = None
    ) -> None:
        """
        Update blur settings at runtime.

        :param blur_type: New blur type (if provided)
        :param blur_strength: New blur strength (if provided)
        """
        if blur_type is not None:
            self.config.blur.type = BlurType(blur_type)
        if blur_strength is not None:
            self.config.blur.strength = blur_strength

        # Update the blurrer instance
        self.blurrer.set_blur_settings(blur_type, blur_strength)

    def _apply_logging_preferences(self) -> None:
        """Align component loggers with configuration preferences."""
        level_name = (self.config.log_level or "INFO").upper()
        if self.config.debug:
            level_name = "DEBUG"
        numeric_level = getattr(logging, level_name, logging.INFO)
        target_loggers = {
            "obscuro": numeric_level,
            "obscuro.detection": numeric_level,
            "obscuro.tracking": numeric_level,
            "obscuro.api": numeric_level,
            __name__: numeric_level,
        }
        for name, level in target_loggers.items():
            logging.getLogger(name).setLevel(level)
        root_logger = logging.getLogger()
        if root_logger.level > numeric_level:
            root_logger.setLevel(numeric_level)

    def update_detection_thresholds(
        self,
        confidence_threshold: float | None = None,
        low_score_threshold: float | None = None,
    ) -> None:
        """
        Update detection thresholds at runtime.

        :param confidence_threshold: New global detection threshold (if provided)
        :param low_score_threshold: Minimum score to retain before NMS (if provided)
        """
        if confidence_threshold is not None:
            self.config.detection.confidence_threshold = confidence_threshold
        if low_score_threshold is not None:
            self.config.detection.low_score_threshold = low_score_threshold

        # Update the detector instance
        self.detector.set_thresholds(confidence_threshold, low_score_threshold)

        # Update tracker thresholds if tracker supports them
        if hasattr(self.tracker, "set_thresholds"):
            self.tracker.set_thresholds(confidence_threshold, low_score_threshold)

    def update_tracking_settings(
        self,
        tracker_type: str | None = None,
        params: dict[str, float | int | str | bool] | None = None,
    ) -> None:
        """Update tracking type or parameters at runtime."""

        if tracker_type is not None:
            tracker_enum = TrackerType(tracker_type)
            self.config.tracking.type = tracker_enum

        if params:
            self.config.tracking.update_params(params)

        tracker_kwargs = self.config.get_tracker_kwargs()

        if tracker_type is not None:
            self.tracker = TrackerFactory.get(
                name=self.config.tracking.type,
                video_source=self.tracker.video_source,
                cancel_event=self.cancel_event,
                progress_callback=self.progress_callback,
                confidence_threshold=self.config.detection.confidence_threshold,
                low_score_threshold=self.config.detection.low_score_threshold,
                **tracker_kwargs,
            )
        else:
            tracker_params = tracker_kwargs.get("params")
            if isinstance(tracker_params, TrackerParams):
                self.tracker.reconfigure(
                    tracker_params,
                    confidence_threshold=self.config.detection.confidence_threshold,
                    low_score_threshold=self.config.detection.low_score_threshold,
                )

    def get_tracker_info(self) -> dict[str, Any]:
        """
        Get information about the current tracker.

        :return: Dictionary with tracker information
        """
        return self.tracker.get_tracker_info()  # type: ignore

    def _apply_offline_linker_if_enabled(
        self,
        tracks: pl.DataFrame,
        video_path: Path | None,
    ) -> pl.DataFrame:
        """Optionally run the offline linker and update debug metadata."""
        should_emit_debug_plot = bool(os.environ.get("BLUR_DEBUG_TRACK_PLOT"))

        descriptor = getattr(type(self.tracker), "track_history", None)
        timeline_property = isinstance(descriptor, property)

        timeline = getattr(self.tracker, "track_history", None)
        if timeline is None:
            return tracks

        if isinstance(timeline, list):
            history = list(timeline)
        else:
            try:
                history = list(timeline)
            except TypeError:
                return tracks

        if not history:
            return tracks

        pre_plot_data = (
            self._collect_track_centers_snapshot(history) if should_emit_debug_plot else []
        )

        if not any(
            hasattr(obs, "track_id") and hasattr(obs, "frame") and hasattr(obs, "tlwh")
            for obs in history
        ):
            return tracks

        params = self.config.tracking.effective_params()
        video_id = str(video_path) if video_path else None
        mapping, filled_tracks = link_tracklets(video_id, history, params)
        self._offline_linker_tracks = filled_tracks
        if mapping:

            def remap_track_id(value: object) -> object:
                try:
                    key = int(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return value
                return mapping.get(key, key)

            if not tracks.is_empty() and "track_id" in tracks.columns:
                tracks = tracks.with_columns(
                    pl.col("track_id")
                    .map_elements(remap_track_id, return_dtype=pl.Int64)
                    .alias("track_id")
                )

            for obs in history:
                if not hasattr(obs, "track_id"):
                    continue
                try:
                    original_id = int(obs.track_id)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                new_id = mapping.get(original_id, original_id)
                if new_id != original_id and hasattr(obs, "debug_color"):
                    obs.debug_color = OFFLINE_LINK_DEBUG_COLOR
                obs.track_id = new_id

            if hasattr(self.tracker, "_timeline") and isinstance(self.tracker._timeline, list):
                self.tracker._timeline = history  # type: ignore[attr-defined]
            elif not timeline_property:
                with contextlib.suppress(AttributeError):
                    self.tracker.track_history = history

        if should_emit_debug_plot:
            post_plot_data = self._collect_track_centers_snapshot(history)
            self._maybe_write_tracking_debug_plot(video_path, pre_plot_data, post_plot_data)

        return tracks

    def _collect_track_centers_snapshot(self, history: list) -> list[dict[str, float]]:
        """Extract normalized center positions for plotting."""
        snapshot: list[dict[str, float]] = []
        for obs in history:
            track_id = getattr(obs, "track_id", None)
            frame = getattr(obs, "frame", None)
            tlwh = getattr(obs, "tlwh", None)
            frame_size = getattr(obs, "frame_size", None)
            if track_id is None or frame is None or tlwh is None:
                continue
            center = normalized_center(tlwh, frame_size)
            if center is None:
                continue
            try:
                frame_idx = int(frame)
            except (TypeError, ValueError):
                continue
            try:
                track_key = str(int(track_id))
            except (TypeError, ValueError):
                track_key = str(track_id)
            snapshot.append(
                {
                    "track_key": track_key,
                    "frame": frame_idx,
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                }
            )
        return snapshot

    def _maybe_write_tracking_debug_plot(
        self,
        video_path: Path | str | None,
        pre_data: list[dict[str, float]],
        post_data: list[dict[str, float]],
    ) -> None:
        """Emit a Plotly plot of track centers when BLUR_DEBUG_TRACK_PLOT is set."""
        if not pre_data and not post_data:
            return
        if video_path is None:
            return

        from plotly import graph_objects as go  # type: ignore
        from plotly.colors import qualitative  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore

        output_path = Path(video_path)
        output_file = output_path.with_name(f"{output_path.stem}_tracks.html")

        color_palette = (
            list(qualitative.Plotly)
            + list(getattr(qualitative, "Safe", []))
            + list(getattr(qualitative, "Dark24", []))
            + list(getattr(qualitative, "Light24", []))
        )
        if not color_palette:
            color_palette = ["#1f77b4"]

        color_map: dict[str, str] = {}

        def assign_color(track_key: str) -> str:
            if track_key not in color_map:
                color_map[track_key] = color_palette[len(color_map) % len(color_palette)]
            return color_map[track_key]

        for dataset in (pre_data, post_data):
            for row in dataset:
                assign_color(row["track_key"])

        figure = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            subplot_titles=("Center X vs Frame", "Center Y vs Frame"),
        )

        stages: list[tuple[str, list[dict[str, float]], str]] = [
            ("pre-link", pre_data, "dot"),
            ("post-link", post_data, "solid"),
        ]

        for stage_name, data, dash in stages:
            if not data:
                continue
            tracks: dict[str, list[dict[str, float]]] = {}
            for row in data:
                tracks.setdefault(row["track_key"], []).append(row)
            for track_key, entries in tracks.items():
                entries.sort(key=lambda item: item["frame"])
                frames = [item["frame"] for item in entries]
                centers_x = [item["center_x"] for item in entries]
                centers_y = [item["center_y"] for item in entries]
                color = assign_color(track_key)
                hover_base = (
                    f"stage={stage_name}<br>track={track_key}<br>frame=%{{y}}<br>center=%{{x:.3f}}"
                )
                figure.add_trace(
                    go.Scatter(
                        x=centers_x,
                        y=frames,
                        mode="lines+markers",
                        name=f"{stage_name} · track {track_key} (x)",
                        line={"color": color, "dash": dash, "width": 2},
                        marker={"size": 6, "color": color},
                        legendgroup=f"{stage_name}-{track_key}-x",
                        hovertemplate=hover_base + "<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                figure.add_trace(
                    go.Scatter(
                        x=centers_y,
                        y=frames,
                        mode="lines+markers",
                        name=f"{stage_name} · track {track_key} (y)",
                        line={"color": color, "dash": dash, "width": 2},
                        marker={"size": 6, "color": color},
                        legendgroup=f"{stage_name}-{track_key}-y",
                        showlegend=False,
                        hovertemplate=hover_base.replace("center", "center_y") + "<extra></extra>",
                    ),
                    row=1,
                    col=2,
                )

        figure.update_xaxes(title_text="Normalized center X", row=1, col=1, range=[0.0, 1.0])
        figure.update_xaxes(title_text="Normalized center Y", row=1, col=2, range=[0.0, 1.0])
        figure.update_yaxes(title_text="Frame index", row=1, col=1)
        figure.update_layout(
            title=f"Track centers before/after offline linking · {output_path.name}",
            template="plotly_white",
            height=700,
            hovermode="closest",
        )

        try:
            figure.write_html(str(output_file), include_plotlyjs="cdn", full_html=True)
            logger.info("Wrote tracking debug plot to %s", output_file)
        except Exception as exc:  # pragma: no cover - debug utility
            logger.warning("Failed to write tracking debug plot %s: %s", output_file, exc)

    def blur_video(self, input_path: Path, output_path: Path) -> None:
        """
        Orchestrate the three-step anonymization pipeline: detect, track, blur.
        Each stage reports 0-100% progress independently.
        """
        try:
            # Set the video source for tracking
            self.tracker.set_video_source(input_path)

            # Step 1: Detection
            self.safe_progress_update(0, "Detection", "Starting")
            detections = self._detect(input_path)
            self.safe_progress_update(30, "Detection", "Complete")

            # Step 2: Tracking
            self.safe_progress_update(30, "Tracking", "Starting")
            tracks = self.tracker.track(detections)
            tracks = self._apply_offline_linker_if_enabled(
                tracks, getattr(self.tracker, "video_source", input_path)
            )
            self.safe_progress_update(60, "Tracking", "Complete")

            # Step 3: Blurring
            self.safe_progress_update(60, "Blurring", "Starting")
            use_debug_overlay = (
                getattr(self.config.blur, "type", BlurType.GAUSSIAN) == BlurType.DEBUG
            )
            if use_debug_overlay:
                debug_rows = [obs.as_dict(include_debug=True) for obs in self.tracker.track_history]
                debug_tracks = pl.DataFrame(debug_rows) if debug_rows else pl.DataFrame([])
                self.blurrer.blur_video(
                    input_path,
                    debug_tracks,
                    output_path,
                    codec=self.config.video.codec,
                    quality=self.config.video.quality,
                    raw_detections=detections,
                )
            else:
                self.blurrer.blur_video(
                    input_path,
                    tracks,
                    output_path,
                    codec=self.config.video.codec,
                    quality=self.config.video.quality,
                )
            self.safe_progress_update(100, "Blurring", "Complete")

            self.safe_progress_update(100, "Processing", "Complete")

        except CancellationException:
            # Clean up partial files
            if output_path.exists():
                output_path.unlink()
            raise
        except Exception as e:
            print(f"Error processing video: {e}")
        finally:
            self.detector.clear_mask_cache()

    def _detect(self, input: NDArrayUint8 | Path) -> pl.DataFrame:
        """
        Delegate detection to the Detector instance.
        Supports both file paths and numpy arrays.

        :param input: Input image/video as Path or numpy array
        :return: Detection results as Polars DataFrame
        """
        return self.detector.detect(input)

    def _update_progress(self, percentage: int, stage: str, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(percentage, stage, message)

    def _is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def blur_image_file(self, input_path: Path, output_path: Path) -> None:
        """
        Blur an image file.

        :param input_path: Path to input image
        :param output_path: Path for output image
        """
        detections = self._detect(input_path)
        self.blurrer.blur_image_file(input_path, detections, output_path)
        self.detector.clear_mask_cache()

    def blur_image_array(self, image: NDArrayUint8) -> NDArrayUint8:
        """
        Blur an image array.

        :param image: Input image as numpy array
        :return: Blurred image as numpy array
        """
        detections = self._detect(image)
        result = self.blurrer.blur_image(image, detections)
        self.detector.clear_mask_cache()
        return result

    def blur_image_arrays(self, images: list[NDArrayUint8]) -> list[NDArrayUint8]:
        """
        Blur multiple image arrays.

        :param images: List of input images as numpy arrays
        :return: List of blurred images as numpy arrays
        """
        blurred_images = []
        total_images = len(images)

        for i, image in enumerate(images):
            if self._is_cancelled():
                break

            self._update_progress(
                int((i / total_images) * 100),
                "Processing",
                f"Processing image {i + 1}/{total_images}",
            )

            blurred_image = self.blur_image_array(image)
            blurred_images.append(blurred_image)

        return blurred_images

    def process_mixed_inputs(
        self, inputs: list[NDArrayUint8 | Path], output_paths: list[Path] | None = None
    ) -> list[NDArrayUint8 | None]:
        """
        Process a mixed list of inputs (arrays and file paths).

        :param inputs: List of inputs (numpy arrays or file paths)
        :param output_paths: Optional list of output paths for file inputs
        :return: List of results (numpy arrays for array inputs, None for file inputs)
        """
        results = []
        total_inputs = len(inputs)

        for i, input_item in enumerate(inputs):
            if self._is_cancelled():
                break

            self._update_progress(
                int((i / total_inputs) * 100),
                "Processing",
                f"Processing input {i + 1}/{total_inputs}",
            )

            if isinstance(input_item, np.ndarray):
                # Process array input
                result = self.blur_image_array(input_item)
                results.append(result)
            elif isinstance(input_item, Path):
                # Process file input
                if output_paths and i < len(output_paths):
                    self.blur_image_file(input_item, output_paths[i])
                results.append(None)
            else:
                raise ValueError(f"Unsupported input type: {type(input_item)}")

        return results

    def get_current_settings(self) -> dict[str, Any]:
        """
        Get current anonymizer settings.

        :return: Dictionary with current settings
        """
        return {
            "model_path": str(self.config.model.path),
            "blur_type": self.config.blur.type.value,
            "blur_strength": self.config.blur.strength,
            "available_blur_types": self.get_available_blur_types(),
            "tracker_info": self.get_tracker_info(),
        }

    def detect_image(self, input: NDArrayUint8 | Path) -> pl.DataFrame:
        """
        Public interface for detection only.

        :param input: Input as numpy array or Path
        :return: Detection results as Polars DataFrame
        """
        return self._detect(input)

    def set_runtime_hooks(
        self,
        *,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[int, str, str], None] | None = None,
    ) -> None:
        """Update cancellation and progress callbacks across all components."""

        if cancel_event is not None:
            self.cancel_event = cancel_event
            self.detector.cancel_event = cancel_event
            self.blurrer.cancel_event = cancel_event
            if hasattr(self.tracker, "cancel_event"):
                self.tracker.cancel_event = cancel_event

        if progress_callback is not None:
            self._raw_progress_callback = progress_callback
            throttled = throttle_progress_callback(
                progress_callback,
                interval_seconds=self._progress_interval_seconds,
                percent_step=self._progress_percent_step,
            )
            self.progress_callback = throttled
            self.detector.progress_callback = throttled
            self.blurrer.progress_callback = throttled
            if hasattr(self.tracker, "progress_callback"):
                self.tracker.progress_callback = throttled
