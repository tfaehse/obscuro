#!/usr/bin/env python3
"""
Simple CLI for the blur anonymization tool.
Provides command-line access to the same functionality as the API.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import imageio.v3 as iio

from anonymizer import (
    Anonymizer,
    AnonymizerConfig,
    create_config_template,
    load_config,
)
from anonymizer.config import (
    BlurType,
    TrackerType,
    enforce_model_batch_constraints,
    merge_config,
    set_config,
)
from anonymizer.paths import get_detection_models_dir
from anonymizer.tempfile_cleanup import cleanup_orphaned_temp_dirs
from blur_cli.logging_setup import get_progress_logger, log_with_extra, setup_logging

logger = logging.getLogger("obscuro.cli")


def apply_model_constraints_with_notice(config: AnonymizerConfig, logger: logging.Logger) -> None:
    """Ensure config honors model-specific limits and inform the user."""
    model_name = getattr(config.model, "name", None)
    static_batch = bool(model_name and model_name.endswith("_b1"))
    changed = enforce_model_batch_constraints(config, log=logger)
    if static_batch and not changed:
        logger.info(
            "Model %s enforces batch size 1 (CoreML export). Batch size remains fixed.",
            model_name,
        )


def blur_image(args):
    """Blur detected objects (bike, head, person, plate, vehicle) in a single image."""
    progress_logger = get_progress_logger()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist")
        return 1

    if not input_path.is_file():
        logger.error(f"{input_path} is not a file")
        return 1

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output filename
        output_path = input_path.parent / f"{input_path.stem}_blurred{input_path.suffix}"

    logger.info(f"Processing image: {input_path}")
    logger.info(f"Output will be saved to: {output_path}")

    # Progress callback for CLI
    def progress_callback(percent, stage, message):
        progress_logger.info(f"Progress: {percent:.2f}% - {stage}: {message}")

    try:
        # Load configuration and apply CLI overrides
        config = get_config_for_args(args)
        apply_model_constraints_with_notice(config, logger)

        log_with_extra(
            logger,
            "debug",
            "Configuration loaded",
            input_file=str(input_path),
            output_file=str(output_path),
            model_name=config.model.name,
            blur_type=config.blur.type.value,
            inference_size=config.detection.inference_size,
        )

        # Initialize anonymizer with configuration
        anonymizer = Anonymizer(
            config=config,
            progress_callback=progress_callback,
        )

        # Load and process image (RGB)
        try:
            image = iio.imread(input_path)
        except (OSError, ValueError) as exc:
            logger.error("Could not load image %s: %s", input_path, exc)
            return 1

        logger.info("Detecting and blurring...")
        result = anonymizer.blur_image_array(image)

        # Save result
        try:
            iio.imwrite(output_path, result)
        except (OSError, ValueError):
            logger.error("Could not save image to %s", output_path)
            return 1

        logger.info(f"Successfully saved blurred image to: {output_path}")
        log_with_extra(
            logger,
            "info",
            "Image processing completed successfully",
            processing_time_seconds=None,  # Could add timing info
            input_size=f"{image.shape[1]}x{image.shape[0]}",
            output_file=str(output_path),
        )
        return 0

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        return 1


def blur_video(args):
    """Blur detected objects (bike, head, person, plate, vehicle) in a video."""
    progress_logger = get_progress_logger()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file {input_path} does not exist")
        return 1

    if not input_path.is_file():
        logger.error(f"{input_path} is not a file")
        return 1

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output filename
        output_path = input_path.parent / f"{input_path.stem}_blurred{input_path.suffix}"

    logger.info(f"Processing video: {input_path}")
    logger.info(f"Output will be saved to: {output_path}")

    # Progress callback for CLI
    def progress_callback(percent, stage, message):
        progress_logger.info(f"Progress: {percent:.2f}% - {stage}: {message}")

    try:
        # Load configuration and apply CLI overrides
        config = get_config_for_args(args)
        apply_model_constraints_with_notice(config, logger)

        log_with_extra(
            logger,
            "debug",
            "Configuration loaded for video processing",
            input_file=str(input_path),
            output_file=str(output_path),
            model_name=config.model.name,
            blur_type=config.blur.type.value,
            tracker_type=config.tracking.type.value,
            inference_size=config.detection.inference_size,
        )

        # Initialize anonymizer with configuration
        anonymizer = Anonymizer(
            config=config,
            progress_callback=progress_callback,
        )

        logger.info("Starting video processing...")
        ret = anonymizer.blur_video(input_path, output_path)

        if ret == 0:
            logger.info(f"Successfully saved blurred video to: {output_path}")
            log_with_extra(
                logger,
                "info",
                "Video processing completed successfully",
                input_file=str(input_path),
                output_file=str(output_path),
            )
        return ret

    except Exception as e:
        logger.error(f"CLI: Error processing video: {e}", exc_info=True)
        return 1


# Mapping from CLI argument names to config paths (dotted notation)
CLI_TO_CONFIG_MAP: dict[str, str] = {
    "blur_type": "blur.type",
    "blur_strength": "blur.strength",
    "confidence_threshold": "detection.confidence_threshold",
    "low_score_threshold": "detection.low_score_threshold",
    "batch_size": "detection.batch_size",
    "inference_size": "detection.inference_size",
    "sahi_overlap": "detection.sahi_overlap_ratio",
    "single_pass": "detection.single_pass",
    "disable_masks": "detection.disable_masks",
    "tracker": "tracking.type",
    "bidirectional": "tracking.bidirectional_mode",
    "bidirectional_base_tracker": "tracking.bidirectional_base_tracker",
    "offline_linker": "tracking.use_offline_linker",
    "video_codec": "video.codec",
    "video_quality": "video.quality",
    "config_debug": "debug",
    "log_level": "log_level",
}


def _set_nested(target: dict[str, Any], path: str, value: Any) -> None:
    """Set a value in a nested dict using dotted path."""
    parts = path.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def get_config_for_args(args) -> AnonymizerConfig:
    """Create configuration from command line arguments."""
    overrides: dict[str, Any] = {}

    # Handle model argument specially (path vs name detection)
    if hasattr(args, "model") and args.model:
        value = args.model.strip()
        candidate = Path(value)
        looks_like_path = (
            candidate.is_absolute()
            or candidate.suffix.lower() == ".onnx"
            or any(sep in value for sep in ("/", "\\"))
        )
        if looks_like_path:
            _set_nested(overrides, "model.file", candidate)
            _set_nested(overrides, "model.name", candidate.stem or None)
        else:
            _set_nested(overrides, "model.name", value)
            _set_nested(overrides, "model.file", None)

    # Process standard mappings
    for attr, config_path in CLI_TO_CONFIG_MAP.items():
        value = getattr(args, attr, None)
        if value is None:
            continue
        # Special type conversions
        if attr == "blur_type":
            value = BlurType(value)
        elif attr == "tracker":
            value = TrackerType(value)
        elif attr in ("batch_size", "inference_size", "video_quality"):
            value = max(1, int(value))
        elif attr in ("single_pass", "offline_linker", "config_debug", "disable_masks"):
            value = bool(value)
        elif attr == "log_level":
            value = str(value).upper()
        _set_nested(overrides, config_path, value)

    # Handle blur_classes (CSV parsing)
    blur_classes_arg = getattr(args, "blur_classes", None)
    if blur_classes_arg:
        if isinstance(blur_classes_arg, str):
            classes = [cls.strip() for cls in blur_classes_arg.split(",") if cls.strip()]
        elif isinstance(blur_classes_arg, list | tuple):
            classes = [str(cls).strip() for cls in blur_classes_arg if str(cls).strip()]
        else:
            classes = [str(blur_classes_arg).strip()]
        _set_nested(overrides, "detection.classes_to_blur", classes)

    # Handle embedding_similarity_gate (goes into tracking.params)
    emb_gate = getattr(args, "embedding_similarity_gate", None)
    if emb_gate is not None and isinstance(emb_gate, int | float | str):
        _set_nested(overrides, "tracking.params.embedding_similarity_gate", float(emb_gate))

    # Handle tracker_params (JSON object)
    if hasattr(args, "tracker_params") and args.tracker_params:
        try:
            user_params = json.loads(args.tracker_params)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tracker params JSON: {exc}") from exc
        if not isinstance(user_params, dict):
            raise ValueError("Tracker params must be a JSON object")
        for key, value in user_params.items():
            _set_nested(overrides, f"tracking.params.{key}", value)

    # Load base configuration and apply overrides
    config_path = getattr(args, "config", None)
    base_config = load_config(config_path=config_path, overrides=None, apply=False)
    resolved = merge_config(base_config, overrides)
    set_config(resolved)
    return resolved


def create_config(args):
    """Create a configuration template file."""

    output_path = Path(args.output) if args.output else Path("blur_config.toml")

    try:
        create_config_template(output_path)
        logger.info(f"Configuration template created at: {output_path}")
        logger.info("You can now edit this file and use it with --config option.")

        log_with_extra(
            logger, "info", "Configuration template created", config_file=str(output_path)
        )
        return 0
    except Exception as e:
        logger.error(f"Error creating config template: {e}", exc_info=True)
        return 1


def _get_models_dir() -> Path:
    # Allow callers to observe missing directories without eagerly copying bundled models.
    return get_detection_models_dir(create=False)


def list_models(download_url: str | None = None, desired_name: str | None = None):
    """List available models, optionally downloading a new checkpoint."""

    models_path = _get_models_dir()
    if download_url:
        models_path.mkdir(parents=True, exist_ok=True)
        try:
            logger.info(f"Downloading model from {download_url}")
            parsed = urlparse(download_url)
            if parsed.scheme not in {"http", "https"}:
                logger.error("Only http and https download URLs are supported")
                return 1
            with urlopen(download_url) as response:  # nosec: B310 (scheme validated above)
                data = response.read()
        except (URLError, HTTPError, ValueError) as exc:  # pragma: no cover - network exceptions
            logger.error("Failed to download model: %s", exc)
            return 1

        if not data:
            logger.error("Downloaded file is empty")
            return 1

        default_name = Path(parsed.path).stem or "model"
        target_stem = desired_name or default_name
        target_stem = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in target_stem
        ).strip("_")
        if not target_stem:
            target_stem = "model"
        target_path = models_path / f"{target_stem}.onnx"
        counter = 0
        while target_path.exists():
            counter += 1
            target_path = models_path / f"{target_stem}_{counter}.onnx"

        try:
            with target_path.open("wb") as handle:
                handle.write(data)
            logger.info(f"Saved model to {target_path}")
        except OSError as exc:
            logger.error(f"Failed to write model: {exc}")
            return 1

    if not models_path.exists():
        logger.error("No models directory found")
        return 1

    try:
        onnx_files = list(models_path.glob("*.onnx"))
    except Exception:
        logger.error("Failed to enumerate models directory")
        return 1
    if not onnx_files:
        logger.warning("No ONNX models found in %s", models_path)
        return 1

    logger.info("Available models in %s:", models_path)
    for model_file in sorted(onnx_files):
        model_name = model_file.stem
        size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info(f"  - {model_name} ({size_mb:.1f} MB)")

    log_with_extra(
        logger,
        "info",
        "Models listed",
        models_directory=str(models_path),
        model_count=len(onnx_files),
        models=[f.stem for f in sorted(onnx_files)],
        downloaded=bool(download_url),
    )

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Blur detected objects (bike, head, person, plate, vehicle) in images and videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Blur an image with default settings
  python cli.py image input.jpg

  # Use a config file
  python cli.py --config blur_config.toml image input.jpg

  # Blur an image with custom output and stronger blur
  python cli.py image input.jpg -o output.jpg --blur-strength 20

  # Blur a video with pixelation instead of gaussian blur
  python cli.py video input.mp4 --blur-type pixelate

  # Create config template
  python cli.py config -o my_config.toml

  # List available models
  python cli.py models

  # Enable JSON logging
  python cli.py --json-log logs/blur_processing.json image input.jpg
        """,
    )

    # Global arguments
    parser.add_argument("--config", help="Configuration file to use (TOML format)")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--json-log", type=Path, help="Enable JSON logging to specified file (optional)"
    )
    parser.add_argument(
        "--no-colors", action="store_true", help="Disable colored output in terminal"
    )
    parser.add_argument(
        "--config-debug",
        dest="config_debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable anonymizer debug mode via configuration",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Image command
    image_parser = subparsers.add_parser("image", help="Blur detections in an image")
    image_parser.add_argument("input", help="Input image file")
    image_parser.add_argument("-o", "--output", help="Output image file (default: auto-generated)")

    # Video command
    video_parser = subparsers.add_parser("video", help="Blur detections in a video")
    video_parser.add_argument("input", help="Input video file")
    video_parser.add_argument("-o", "--output", help="Output video file (default: auto-generated)")

    # Config command
    config_parser = subparsers.add_parser("config", help="Create configuration template")
    config_parser.add_argument(
        "-o", "--output", help="Output config file (default: blur_config.toml)"
    )

    # Models command
    models_parser = subparsers.add_parser("models", help="Manage available models")
    models_parser.add_argument("--download", help="Download ONNX model from URL")
    models_parser.add_argument("--name", help="Optional model name (without extension)")

    # Common arguments for image and video
    for subparser in [image_parser, video_parser]:
        subparser.add_argument("--model", help="Model to use (overrides config file)")
        subparser.add_argument(
            "--blur-type",
            choices=["gaussian", "pixelate", "blackout", "debug"],
            help="Type of blur to apply (overrides config file)",
        )
        subparser.add_argument(
            "--blur-classes",
            help="Comma-separated detector classes to blur (e.g., 'plate,head,person')",
        )
        subparser.add_argument(
            "--embedding-similarity-gate",
            type=float,
            dest="embedding_similarity_gate",
            help="Minimum cosine similarity for tracker embeddings (overrides config)",
        )
        subparser.add_argument(
            "--blur-strength", type=int, help="Blur strength/intensity (overrides config file)"
        )

        # Detection arguments
        subparser.add_argument(
            "--disable-masks",
            action="store_true",
            help="Disable segmentation masks (use bounding boxes only)",
        )
        subparser.add_argument(
            "--confidence-threshold",
            dest="confidence_threshold",
            type=float,
            help="Global detection threshold (overrides config file)",
        )
        subparser.add_argument(
            "--low-score-threshold",
            dest="low_score_threshold",
            type=float,
            help="Minimum score retained before NMS (overrides config file)",
        )
        subparser.add_argument(
            "--batch-size",
            type=int,
            dest="batch_size",
            help="Detection batch size (frames per inference call)",
        )
        subparser.add_argument(
            "--inference-size",
            type=int,
            help="Longest image edge (pixels) used for detection inference",
        )
        subparser.add_argument(
            "--sahi-overlap",
            type=float,
            help="SAHI tile overlap ratio between 0.0 and 0.99 (overrides config file)",
        )
        subparser.add_argument(
            "--single-pass",
            dest="single_pass",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Force single-tile SAHI mode (overrides overlap to 0 and inference size to model tile size)",
        )
        subparser.add_argument(
            "--tracker",
            choices=["dummy", "bytetrack", "botsort", "hybrid_sot", "fused", "bidirectional"],
            help="Tracker to use (overrides config file)",
        )
        subparser.add_argument(
            "--tracker-params",
            help="JSON object with tracker parameter overrides (e.g. '{\"distance_gate\":0.15}')",
        )
        subparser.add_argument(
            "--bidirectional",
            dest="bidirectional",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable or disable bidirectional tracking mode (default: disabled)",
        )
        subparser.add_argument(
            "--bidirectional-base-tracker",
            dest="bidirectional_base_tracker",
            choices=["dummy", "bytetrack", "botsort", "hybrid_sot", "fused", "oc_sort"],
            help="Base tracker for bidirectional mode (default: bytetrack)",
        )
        subparser.add_argument(
            "--offline-linker",
            dest="offline_linker",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable or disable the offline tracklet linker (default: enabled)",
        )
        subparser.add_argument(
            "--video-codec",
            dest="video_codec",
            help="Video codec to use for output (overrides config file)",
        )
        subparser.add_argument(
            "--video-quality",
            dest="video_quality",
            type=int,
            help="Video quality (lower is better, overrides config file)",
        )

    args = parser.parse_args()

    # Set up logging based on arguments
    logger = setup_logging(
        log_level=args.log_level, json_log_file=args.json_log, enable_colors=not args.no_colors
    )

    # Clean up orphaned temp directories from crashed processes
    cleanup_orphaned_temp_dirs()

    # Log startup information
    log_with_extra(
        logger,
        "info",
        "CLI started",
        command=args.command,
        log_level=args.log_level,
        json_logging=args.json_log is not None,
    )

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "image":
        return blur_image(args)
    elif args.command == "video":
        return blur_video(args)
    elif args.command == "config":
        return create_config(args)
    elif args.command == "models":
        return list_models(download_url=args.download, desired_name=args.name)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    main()
