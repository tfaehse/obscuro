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
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2

from anonymizer import (
    Anonymizer,
    AnonymizerConfig,
    create_config_template,
    load_config,
)
from anonymizer.config import (
    BlurType,
    ConfigLayers,
    TrackerType,
    enforce_model_batch_constraints,
    set_config,
)
from anonymizer.paths import get_detection_models_dir
from blur_cli.logging_setup import get_progress_logger, log_with_extra, setup_logging


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
    """Blur faces and license plates in a single image."""
    logger = logging.getLogger("obscuro.cli")
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
        progress_logger.info(f"Progress: {percent}% - {stage}: {message}")

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
            use_sahi=config.detection.use_sahi,
            inference_size=config.detection.inference_size,
        )

        # Initialize anonymizer with configuration
        anonymizer = Anonymizer(
            config=config,
            progress_callback=progress_callback,
        )

        # Load and process image
        image = cv2.imread(str(input_path))
        if image is None:
            logger.error(f"Could not load image {input_path}")
            return 1

        logger.info("Detecting and blurring...")
        result = anonymizer.blur_image_array(image)

        # Save result
        success = cv2.imwrite(str(output_path), result)
        if not success:
            logger.error(f"Could not save image to {output_path}")
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
    """Blur faces and license plates in a video."""
    logger = logging.getLogger("obscuro.cli")
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
        progress_logger.info(f"Progress: {percent:3.1f}% - {stage}: {message}")

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
            use_sahi=config.detection.use_sahi,
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


def get_config_for_args(args) -> AnonymizerConfig:
    """Create configuration from command line arguments."""
    override_tree: dict[str, Any] = {}

    def merge(target: dict[str, Any], update: dict[str, Any]) -> None:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    # Model overrides
    if hasattr(args, "model") and args.model:
        value = args.model.strip()
        candidate = Path(value)
        looks_like_path = (
            candidate.is_absolute()
            or candidate.suffix.lower() == ".onnx"
            or any(sep in value for sep in ("/", "\\"))
        )
        model_update: dict[str, Any]
        if looks_like_path:
            model_update = {"model": {"file": candidate, "name": candidate.stem or None}}
        else:
            model_update = {"model": {"name": value, "file": None}}
        merge(override_tree, model_update)

    blur_overrides: dict[str, Any] = {}
    if hasattr(args, "blur_type") and args.blur_type:
        blur_overrides["type"] = BlurType(args.blur_type)
    if hasattr(args, "blur_strength") and args.blur_strength:
        blur_overrides["strength"] = args.blur_strength
    if blur_overrides:
        merge(override_tree, {"blur": blur_overrides})

    detection_overrides: dict[str, Any] = {}
    confidence_arg = getattr(args, "confidence_threshold", None)
    low_score_arg = getattr(args, "low_score_threshold", None)
    if confidence_arg is not None:
        detection_overrides["confidence_threshold"] = confidence_arg
    if low_score_arg is not None:
        detection_overrides["low_score_threshold"] = low_score_arg
    if hasattr(args, "batch_size") and args.batch_size is not None:
        detection_overrides["batch_size"] = max(1, int(args.batch_size))
    if hasattr(args, "use_sahi") and args.use_sahi is not None:
        detection_overrides["use_sahi"] = bool(args.use_sahi)
    if hasattr(args, "inference_size") and args.inference_size is not None:
        detection_overrides["inference_size"] = max(256, int(args.inference_size))
    if hasattr(args, "sahi_overlap") and args.sahi_overlap is not None:
        detection_overrides["sahi_overlap_ratio"] = args.sahi_overlap
    blur_classes_arg = getattr(args, "blur_classes", None)
    if blur_classes_arg:
        if isinstance(blur_classes_arg, str):
            classes = [cls.strip() for cls in blur_classes_arg.split(",") if cls.strip()]
        elif isinstance(blur_classes_arg, list | tuple):
            classes = [str(cls).strip() for cls in blur_classes_arg if str(cls).strip()]
        else:
            classes = [str(blur_classes_arg).strip()]
        detection_overrides["classes_to_blur"] = classes
    if detection_overrides:
        merge(override_tree, {"detection": detection_overrides})

    tracking_overrides: dict[str, Any] = {}
    if hasattr(args, "tracker") and args.tracker:
        tracking_overrides["type"] = TrackerType(args.tracker)
    if hasattr(args, "offline_linker") and args.offline_linker is not None:
        tracking_overrides["use_offline_linker"] = bool(args.offline_linker)
    emb_gate = getattr(args, "embedding_similarity_gate", None)
    if emb_gate is not None and isinstance(emb_gate, int | float | str):
        tracking_overrides.setdefault("params", {})["embedding_similarity_gate"] = float(emb_gate)
    if hasattr(args, "tracker_params") and args.tracker_params:
        try:
            user_params = json.loads(args.tracker_params)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tracker params JSON: {exc}") from exc
        if not isinstance(user_params, dict):
            raise ValueError("Tracker params must be a JSON object")
        tracking_overrides.setdefault("params", {}).update(user_params)
    if tracking_overrides:
        merge(override_tree, {"tracking": tracking_overrides})

    video_overrides: dict[str, Any] = {}
    if hasattr(args, "video_codec") and args.video_codec:
        video_overrides["codec"] = args.video_codec
    if hasattr(args, "video_quality") and args.video_quality is not None:
        video_overrides["quality"] = max(1, int(args.video_quality))
    if video_overrides:
        merge(override_tree, {"video": video_overrides})

    global_overrides: dict[str, Any] = {}
    if getattr(args, "config_debug", None) is not None:
        global_overrides["debug"] = bool(args.config_debug)
    if hasattr(args, "log_level") and args.log_level:
        global_overrides["log_level"] = str(args.log_level).upper()
    if global_overrides:
        merge(override_tree, global_overrides)

    # Load base configuration and apply overrides in one pass
    config_path = getattr(args, "config", None)
    base_config = load_config(
        config_path=config_path,
        overrides=None,
        apply=False,
    )
    layers = ConfigLayers(base_config)
    if override_tree:
        layers.set_layer("cli", override_tree)
    resolved = layers.resolve()
    set_config(resolved)
    return resolved


def create_config(args):
    """Create a configuration template file."""
    logger = logging.getLogger("obscuro.cli")

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
    logger = logging.getLogger("obscuro.cli")

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
        except Exception as exc:  # pragma: no cover - network exceptions
            logger.error(f"Failed to download model: {exc}")
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
        description="Blur faces and license plates in images and videos",
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
    image_parser = subparsers.add_parser("image", help="Blur faces and plates in an image")
    image_parser.add_argument("input", help="Input image file")
    image_parser.add_argument("-o", "--output", help="Output image file (default: auto-generated)")

    # Video command
    video_parser = subparsers.add_parser("video", help="Blur faces and plates in a video")
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
            help="Comma-separated detector classes to blur (e.g., 'license_plate,face')",
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
            "--use-sahi",
            dest="use_sahi",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable or disable SAHI tiled inference (default: disabled)",
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
            "--tracker",
            choices=["dummy", "bytetrack", "botsort", "hybrid_sot", "fused"],
            help="Tracker to use (overrides config file)",
        )
        subparser.add_argument(
            "--tracker-params",
            help="JSON object with tracker parameter overrides (e.g. '{\"distance_gate\":0.15}')",
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
