import argparse
import asyncio
import atexit
import contextlib
import gc
import io
import json
import logging
import re
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import imageio.v3 as iio
import uvicorn
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask

from anonymizer import Anonymizer, load_config
from anonymizer.cancellation import CancellationException
from anonymizer.config import AnonymizerConfig, ConfigLayers, set_config
from anonymizer.model_metadata import load_model_metadata
from anonymizer.paths import (
    DEFAULT_MODEL_NAME,
    IMMUTABLE_MODEL_NAMES,
    ensure_required_models_present,
    get_detection_models_dir,
    get_temp_dir,
)
from anonymizer.tempfile_cleanup import cleanup_orphaned_temp_dirs
from anonymizer.tracking import TRACKER_FACTORY

logger = logging.getLogger("obscuro.api")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = get_detection_models_dir()
ensure_required_models_present(MODELS_DIR)

_base_config = load_config(apply=False)
config_layers = ConfigLayers(_base_config)
set_config(config_layers.resolve())
global_config = _base_config


def _refresh_global_config_cache() -> None:
    """Propagate mutated base config into the global accessor cache."""
    set_config(config_layers.resolve())


def build_effective_config(
    raw_config_json: str,
    base_config: AnonymizerConfig | None = None,
) -> AnonymizerConfig:
    """Return a new config with provided nested overrides applied."""
    try:
        overrides = json.loads(raw_config_json or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}")

    if not isinstance(overrides, dict):  # hard fail on non-object JSON
        raise HTTPException(status_code=400, detail="Config JSON must be an object")

    reference_config = base_config if base_config is not None else config_layers.resolve()
    base_keys = set(reference_config.model_dump().keys())
    unknown = set(overrides) - base_keys
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown top-level keys: {sorted(unknown)}. Allowed: {sorted(base_keys)}",
        )

    try:
        if base_config is not None:
            temp_layers = ConfigLayers(base_config)
            temp_layers.set_layer("request", overrides)
            return temp_layers.resolve()
        return config_layers.resolve(overrides)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Config validation error: {e}")


router = APIRouter(prefix="/blur", tags=["blur"])

# Global instances for GPU and CPU anonymizers
model_lock = threading.Lock()
gpu_anonymizer_instance = None
cpu_anonymizer_instance = None
active_gpu_model_key = None
active_cpu_model_key = None
cpu_lock = threading.Lock()

# Add these globals to the top of the file
video_jobs: dict[str, dict] = {}
video_jobs_lock = threading.Lock()
cancel_events: dict[str, threading.Event] = {}

TEMP_ROOT = get_temp_dir() / "obscuro_jobs"
SESSION_TEMP_DIR = TEMP_ROOT / f"session_{uuid.uuid4().hex}"


def _ensure_session_temp_dir() -> None:
    with contextlib.suppress(FileExistsError):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    # Clean up orphaned temp directories from crashed processes
    cleanup_orphaned_temp_dirs()

    SESSION_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Opportunistically clean up any stale session directories.
    for path in TEMP_ROOT.iterdir():
        if path == SESSION_TEMP_DIR:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                path.unlink()


def _cleanup_session_dir() -> None:
    if SESSION_TEMP_DIR.exists():
        shutil.rmtree(SESSION_TEMP_DIR, ignore_errors=True)


_ensure_session_temp_dir()
atexit.register(_cleanup_session_dir)


def _new_temp_video_path(job_id: str, suffix: str) -> Path:
    unique = uuid.uuid4().hex
    return SESSION_TEMP_DIR / f"{job_id}_{unique}{suffix}"


def _list_model_files() -> list[dict[str, object]]:
    models: list[dict[str, object]] = []
    for path in sorted(MODELS_DIR.glob("*.onnx")):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        models.append(
            {
                "name": path.stem,
                "filename": path.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                "immutable": path.stem in IMMUTABLE_MODEL_NAMES,
                "metadata": load_model_metadata(path),
            }
        )
    return models


def _ensure_active_model_valid() -> None:
    available = _list_model_files()
    if not available:
        return
    if not global_config.model.name or not any(
        model["name"] == global_config.model.name for model in available
    ):
        preferred = next((m for m in available if m["name"] == DEFAULT_MODEL_NAME), None)
        fallback = preferred or available[0]
        global_config.model.name = fallback["name"]
        global_config.model.file = None
        _refresh_global_config_cache()


# Ensure we start with a valid model selection when available
_ensure_active_model_valid()


def get_model_key(anonymizer_config: AnonymizerConfig) -> str:
    """Generate a cache key; include tracker type so switching tracker reloads."""
    tracking_params = anonymizer_config.tracking.effective_params().model_dump()
    params_key = ",".join(f"{key}={tracking_params[key]}" for key in sorted(tracking_params.keys()))
    blur_classes_key = ",".join(sorted(anonymizer_config.detection.classes_to_blur or []))
    return (
        f"{anonymizer_config.model.name}-"
        f"{anonymizer_config.blur.type.value}-"
        f"{anonymizer_config.blur.strength}-"
        f"{anonymizer_config.tracking.type.value}-"
        f"{int(anonymizer_config.tracking.use_offline_linker)}-"
        f"{params_key}-"
        f"{anonymizer_config.detection.confidence_threshold}-"
        f"{anonymizer_config.detection.low_score_threshold}-"
        f"{anonymizer_config.detection.inference_size}-"
        f"{anonymizer_config.detection.sahi_overlap_ratio}-"
        f"{int(anonymizer_config.detection.single_pass)}-"
        f"{int(anonymizer_config.detection.disable_masks)}-"
        f"{blur_classes_key}"
    )


def get_or_create_anonymizer(
    anonymizer_config: AnonymizerConfig,
    progress_callback: Callable[[int, str, str], None] | None,
    cancel_event: threading.Event | None = None,
) -> Anonymizer:
    """Get or create an anonymizer instance based on configuration."""
    return get_or_create_gpu_anonymizer(anonymizer_config, progress_callback, cancel_event)


def get_or_create_gpu_anonymizer(
    anonymizer_config: AnonymizerConfig,
    progress_callback: Callable[[int, str, str], None] | None,
    cancel_event: threading.Event | None = None,
) -> Anonymizer:
    """Get or create a GPU-backed anonymizer instance based on configuration."""
    global gpu_anonymizer_instance, active_gpu_model_key
    key = get_model_key(anonymizer_config)

    if key != active_gpu_model_key:
        logger.info(f"Loading new GPU model: {key}")
        gpu_anonymizer_instance = Anonymizer(
            config=anonymizer_config,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            execution_providers=None,  # Auto-detect GPU
        )
        active_gpu_model_key = key
    else:
        logger.debug(f"Reusing GPU model: {key}")

    if gpu_anonymizer_instance is None:
        raise RuntimeError("Failed to initialise GPU anonymizer instance")

    instance = gpu_anonymizer_instance
    assert instance is not None
    instance.set_runtime_hooks(cancel_event=cancel_event, progress_callback=progress_callback)

    if cancel_event is not None:
        cancel_event.clear()

    return instance


def get_or_create_cpu_anonymizer(
    anonymizer_config: AnonymizerConfig,
    progress_callback: Callable[[int, str, str], None] | None,
    cancel_event: threading.Event | None = None,
) -> Anonymizer:
    """Get or create a CPU-only anonymizer instance based on configuration."""
    global cpu_anonymizer_instance, active_cpu_model_key
    key = get_model_key(anonymizer_config)

    if key != active_cpu_model_key:
        logger.info(f"Loading new CPU model: {key}")
        cpu_anonymizer_instance = Anonymizer(
            config=anonymizer_config,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            execution_providers=["CPUExecutionProvider"],  # Force CPU
        )
        active_cpu_model_key = key
    else:
        logger.debug(f"Reusing CPU model: {key}")

    if cpu_anonymizer_instance is None:
        raise RuntimeError("Failed to initialise CPU anonymizer instance")

    instance = cpu_anonymizer_instance
    assert instance is not None
    instance.set_runtime_hooks(cancel_event=cancel_event, progress_callback=progress_callback)

    if cancel_event is not None:
        cancel_event.clear()

    return instance


def is_gpu_busy() -> bool:
    """Check if the GPU is currently busy processing a video job."""
    acquired = model_lock.acquire(blocking=False)
    if acquired:
        model_lock.release()
        return False
    return True


def get_backend_health(*, blocking: bool = True) -> dict[str, object]:
    acquired = model_lock.acquire(blocking=blocking)
    if not acquired:
        return {
            "status": "busy",
            "status_code": 1,
            "execution_provider": None,
            "requested_providers": [],
            "active_providers": [],
            "detail": "Anonymizer in use",
        }

    try:
        anonymizer = get_or_create_anonymizer(global_config, progress_callback=None)
        detector_status = anonymizer.detector.get_execution_provider_status()
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "status": "error",
            "status_code": -1,
            "execution_provider": None,
            "requested_providers": [],
            "active_providers": [],
            "detail": str(exc),
        }
    finally:
        model_lock.release()

    status_code = detector_status.get("status_code", -1)
    status_text = "ok" if status_code == 0 else "degraded"
    return {
        "status": status_text,
        "status_code": status_code,
        "execution_provider": detector_status.get("primary"),
        "requested_providers": detector_status.get("requested", []),
        "active_providers": detector_status.get("active", []),
    }


def make_progress_callback(job_id: str):
    def callback(percent: float, stage: str, message: str):
        with video_jobs_lock:
            if job_id in video_jobs:
                video_jobs[job_id]["job_id"] = job_id
                video_jobs[job_id]["progress"] = float(percent)
                video_jobs[job_id]["stage"] = stage
                video_jobs[job_id]["stage_message"] = message
                video_jobs[job_id]["message"] = f"{stage}: {message}"
                sequence = int(video_jobs[job_id].get("sequence", 0)) + 1
                video_jobs[job_id]["sequence"] = sequence
                video_jobs[job_id]["updated_at"] = time.time()

    return callback


@router.get("/config/options")
async def get_config_options():
    """Get available configuration options for the GUI."""
    models = _list_model_files()
    if models:
        _ensure_active_model_valid()

    available_models = [m["name"] for m in models]
    active_model_path = MODELS_DIR / f"{global_config.model.name}.onnx"
    meta = load_model_metadata(active_model_path)
    available_classes = meta.get("classes", [])
    default_blur_classes = meta.get("default_blur", [])
    current_classes = [
        cls for cls in global_config.detection.classes_to_blur if cls in available_classes
    ]
    if not current_classes:
        current_classes = list(default_blur_classes or available_classes)
        global_config.detection.classes_to_blur = list(current_classes)
        _refresh_global_config_cache()

    return {
        "model": {
            "available": available_models,
            "current": global_config.model.name,
            "files": models,
        },
        "blur": {
            "types": ["gaussian", "pixelate", "blackout", "debug"],
            "current_type": global_config.blur.type.value,
            "current_strength": global_config.blur.strength,
            "strength_range": [1, 100],
        },
        "detection": {
            "current_confidence_threshold": global_config.detection.confidence_threshold,
            "current_low_score_threshold": global_config.detection.low_score_threshold,
            "current_batch_size": global_config.detection.batch_size,
            "threshold_range": [0.0, 1.0],
            "current_inference_size": global_config.detection.inference_size,
            "inference_size_range": [256, 8192],
            "current_sahi_overlap": global_config.detection.sahi_overlap_ratio,
            "sahi_overlap_range": [0.0, 0.99],
            "current_single_pass": global_config.detection.single_pass,
            "available_classes": list(available_classes),
            "current_classes": list(current_classes),
            "default_blur_classes": list(default_blur_classes),
            "current_disable_masks": global_config.detection.disable_masks,
        },
        "tracking": {
            "types": sorted(TRACKER_FACTORY.keys()),
            "current_type": global_config.tracking.type.value,
            "params": global_config.tracking.effective_params().model_dump(),
            "use_offline_linker": global_config.tracking.use_offline_linker,
            "ranges": {
                "distance_gate": [0.05, 1.0],
                "confirm_after_N": [1, 5],
                "max_misses_M": [1, 30],
                "bbox_dilate_pct": [0.05, 0.4],
                "temporal_smooth_alpha": [0.0, 1.0],
                "vt_max_age": [0, 30],
                "distance_gate_hi": [0.001, 1.0],
                "distance_gate_lo": [0.001, 1.0],
                "drift_gate": [0.0, 1.0],
            },
        },
        "video": {
            "codecs": ["h264", "hevc", "vp8", "vp9"],
            "current_codec": global_config.video.codec,
            "current_quality": global_config.video.quality,
            "quality_range": [1, 51],
        },
        "global": {
            "log_levels": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "current_debug": global_config.debug,
            "current_log_level": global_config.log_level,
        },
    }


@router.get("/models")
async def list_models():
    """Return metadata about available model checkpoints."""
    return {"models": _list_model_files()}


def _sanitize_model_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return sanitized or "model"


@router.post("/models")
async def upload_model(
    file: UploadFile = File(...),
    name: str | None = Form(None),
):
    """Upload a new ONNX model into the managed models directory."""
    if not file.filename and not name:
        raise HTTPException(status_code=400, detail="Model filename is required")

    original_suffix = Path(file.filename or "").suffix.lower()
    if original_suffix and original_suffix != ".onnx":
        raise HTTPException(status_code=400, detail="Only .onnx models are supported")

    desired_stem = _sanitize_model_name(name or Path(file.filename or "model").stem)

    counter = 0
    target_path = MODELS_DIR / f"{desired_stem}.onnx"
    while target_path.exists():
        counter += 1
        target_path = MODELS_DIR / f"{desired_stem}_{counter}.onnx"

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        with target_path.open("wb") as fh:
            fh.write(content)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write model: {exc}") from exc

    models = _list_model_files()
    _ensure_active_model_valid()
    return {
        "models": models,
        "added": next((m for m in models if m["filename"] == target_path.name), None),
    }


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Remove a managed model checkpoint."""
    candidate = Path(model_name)
    if candidate.suffix != ".onnx":
        candidate = candidate.with_suffix(".onnx")

    # Prevent directory traversal by resolving and checking parent
    target_path = (MODELS_DIR / candidate.name).resolve()
    if target_path.parent != MODELS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid model name")

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    if target_path.stem in IMMUTABLE_MODEL_NAMES:
        raise HTTPException(status_code=403, detail="Built-in models cannot be deleted")

    try:
        target_path.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {exc}") from exc

    models = _list_model_files()
    if models:
        _ensure_active_model_valid()
    else:
        global_config.model.name = None
        global_config.model.file = None
        _refresh_global_config_cache()
    return {"models": models}


@router.get("/config/current")
async def get_current_config():
    """Get the current configuration."""
    return global_config.model_dump()


@router.post("/frame")
async def blur_frame(
    input: UploadFile = File(...),
    config: str = Form("{}"),  # nested or flat override structure
):
    logger.debug("Frame blur request received")
    contents = await input.read()
    try:
        frame = iio.imread(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc

    # Build effective config
    effective_config = build_effective_config(config, global_config)

    # Try GPU first (non-blocking), fall back to CPU if GPU is busy
    if is_gpu_busy():
        logger.debug("GPU busy, using CPU anonymizer for frame preview")
        with cpu_lock:
            anonymizer = get_or_create_cpu_anonymizer(effective_config, None, None)
            result = anonymizer.blur_image_array(frame)
    else:
        logger.debug("GPU available, using GPU anonymizer for frame preview")
        with model_lock:
            anonymizer = get_or_create_gpu_anonymizer(effective_config, None, None)
            result = anonymizer.blur_image_array(frame)

    try:
        encoded_img = iio.imwrite("<bytes>", result, extension=".jpg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to encode image") from exc
    return StreamingResponse(io.BytesIO(encoded_img), media_type="image/jpeg")


@router.post("/image_file")
async def blur_image_file(
    input: Path = Form(...),
    output_path: Path = Form(...),
    config: str = Form("{}"),
):
    effective_config = build_effective_config(config, global_config)

    with model_lock:
        if not input.exists():
            raise HTTPException(status_code=400, detail="Input file does not exist")

        try:
            anonymizer = get_or_create_anonymizer(effective_config, None, None)
            anonymizer.blur_image_file(input, output_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Image blurred and saved successfully."}


@router.post("/video_file")
async def blur_video_async(
    input: UploadFile = File(...),
    config: str = Form("{}"),  # JSON string: partial nested AnonymizerConfig structure
    output_filename: str = Form("output.mp4"),
):
    """Process uploaded video file with blur effects."""
    job_id = str(uuid.uuid4())
    # Build effective config (strict nested override only)
    effective_config = build_effective_config(config, global_config)

    # Create cancel event for this job
    cancel_event = threading.Event()

    # Persist the uploaded payload to a temporary file synchronously; this keeps
    # the request lifecycle simple and allows us to return the job id as soon as
    # the payload is on disk.
    temp_input_path = _new_temp_video_path(job_id, "_input.mp4")

    try:
        with temp_input_path.open("wb") as sink:
            while True:
                chunk = await input.read(4 * 1024 * 1024)
                if not chunk:
                    break
                sink.write(chunk)
    finally:
        await input.close()

    with video_jobs_lock:
        video_jobs[job_id] = {
            "job_id": job_id,
            "progress": 0,
            "message": "Job started",
            "stage": "Initialization",
            "stage_message": "Job started",
            "status": "running",
            "error": None,
            "sequence": 0,
            "updated_at": time.time(),
            "output_path": None,
        }
        cancel_events[job_id] = cancel_event

    def run_blur(temp_input_path: Path) -> None:
        temp_output_path: Path | None = None
        try:
            temp_output_path = _new_temp_video_path(job_id, "_output.mp4")

            with model_lock:
                # Check if job was cancelled while waiting for lock
                if cancel_events.get(job_id) and cancel_events[job_id].is_set():
                    raise CancellationException("Job cancelled before execution")

                callback = make_progress_callback(job_id)
                anonymizer = get_or_create_anonymizer(effective_config, callback, cancel_event)
                anonymizer.blur_video(temp_input_path, temp_output_path)

            with video_jobs_lock:
                job = video_jobs.get(job_id)
                if job is not None:
                    job["status"] = "done"
                    job["output_path"] = str(temp_output_path)
                    job["sequence"] = int(job.get("sequence", 0)) + 1
                    job["updated_at"] = time.time()

        except CancellationException:
            with video_jobs_lock:
                job = video_jobs.get(job_id)
                if job is not None:
                    job["status"] = "cancelled"
                    job["message"] = "Processing cancelled"
                    job["stage"] = "Cancelled"
                    job["stage_message"] = "Processing cancelled"
                    job["sequence"] = int(job.get("sequence", 0)) + 1
                    job["updated_at"] = time.time()
        except Exception as exc:
            logger.exception(f"Job {job_id} failed: {exc}")
            with video_jobs_lock:
                job = video_jobs.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["error"] = str(exc)
                    job["stage"] = "Error"
                    job["stage_message"] = str(exc)
                    job["sequence"] = int(job.get("sequence", 0)) + 1
                    job["updated_at"] = time.time()
        finally:
            with contextlib.suppress(OSError):
                if temp_input_path.exists():
                    temp_input_path.unlink()
            with video_jobs_lock:
                cancel_events.pop(job_id, None)
                job = video_jobs.get(job_id)
                job_status = job.get("status") if job else None

            # Clean up output if failed or cancelled
            if (
                job_status in {"cancelled", "error"}
                and temp_output_path is not None
                and temp_output_path.exists()
            ):
                with contextlib.suppress(OSError):
                    temp_output_path.unlink()

    threading.Thread(target=run_blur, args=(temp_input_path,), daemon=True).start()
    return {"job_id": job_id}


@router.get("/video_progress/{job_id}")
async def stream_video_progress(job_id: str):
    with video_jobs_lock:
        if job_id not in video_jobs:
            raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_sequence: int | None = None
        while True:
            with video_jobs_lock:
                job = video_jobs.get(job_id)

            if job is None:
                break

            sequence = int(job.get("sequence", 0))

            if last_sequence is None or sequence != last_sequence:
                yield {"data": json.dumps(job, default=str)}
                last_sequence = sequence

            if job["status"] in ("done", "error", "cancelled"):
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@router.post("/cancel_video/{job_id}")
async def cancel_video_processing(job_id: str):
    """Cancel a running video processing job."""
    with video_jobs_lock:
        if job_id not in video_jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        if job_id in cancel_events:
            # Set the cancel event to signal cancellation
            cancel_events[job_id].set()

            # Update job status
            video_jobs[job_id]["status"] = "cancelled"
            video_jobs[job_id]["message"] = "Cancelled by user"
            video_jobs[job_id]["stage"] = "Cancelled"
            video_jobs[job_id]["stage_message"] = "Cancelled by user"
            video_jobs[job_id]["sequence"] = int(video_jobs[job_id].get("sequence", 0)) + 1
            video_jobs[job_id]["updated_at"] = time.time()

            return {"message": "Job cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled")


@router.get("/download/{job_id}")
async def download_video_result(job_id: str):
    """Download the processed video result."""
    with video_jobs_lock:
        job = video_jobs.get(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if job["status"] != "done":
            raise HTTPException(status_code=400, detail="Job not completed")

        output_path = job.get("output_path")
        if not output_path or not Path(output_path).exists():
            raise HTTPException(status_code=404, detail="Output file not found")

    def finalize_download() -> None:
        with video_jobs_lock:
            video_jobs.pop(job_id, None)
        with contextlib.suppress(OSError):
            Path(output_path).unlink()

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"blurred_video_{job_id}.mp4",
        background=BackgroundTask(finalize_download),
    )


@router.post("/reset")
async def reset_backend():
    """Forcefully reset the backend state."""
    global gpu_anonymizer_instance, cpu_anonymizer_instance
    global active_gpu_model_key, active_cpu_model_key

    logger.warning("Backend reset requested")

    # 1. Clear all jobs
    with video_jobs_lock:
        video_jobs.clear()
        # Set all cancel events to stop running threads
        for event in cancel_events.values():
            event.set()
        cancel_events.clear()

    # 2. Reset anonymizer instances
    # We try to acquire the lock to safely reset, but if it's held by a stuck thread,
    # we might need to proceed anyway or risk deadlocking the reset itself.
    # For now, we'll try to acquire with a timeout.
    acquired = model_lock.acquire(timeout=2.0)
    if not acquired:
        logger.error("Could not acquire model_lock during reset - forcing reset anyway")
        # If we can't acquire, we assume the lock is held by a stuck thread.
        # Python threading.Lock doesn't support forced release from another thread.
        # We can only reset the global variables and hope the stuck thread eventually dies or fails.
        # In a real production system, this might require process restart.
        # For this app, we'll just reset the globals so new requests create new instances.

    try:
        gpu_anonymizer_instance = None
        cpu_anonymizer_instance = None
        active_gpu_model_key = None
        active_cpu_model_key = None

        # Force garbage collection to help clean up
        gc.collect()

        # Clean up temp directory
        _cleanup_session_dir()
        _ensure_session_temp_dir()

    finally:
        if acquired:
            model_lock.release()
        elif model_lock.locked():
            # If we didn't acquire it but it's locked, we can't release it safely.
            # This is a critical failure state.
            # However, since we reset the instances, new requests will try to create new ones.
            # The `model_lock` is global, so if it's permanently stuck, we are in trouble.
            # A "hard" reset might be needed (restarting the server).
            # For now, let's just log it.
            pass

    return {"message": "Backend reset successfully"}


app.include_router(router)


@app.api_route("/healthz", methods=["GET", "HEAD"])
async def healthz():
    return get_backend_health(blocking=False)


def start_server(
    host="0.0.0.0",
    port=8000,
    log_level="INFO",
    json_log_file=None,
    reload: bool = False,
    config_path: Path | None = None,
):
    """Start the FastAPI server with logging configuration."""
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parents[1]))

    from blur_cli.logging_setup import setup_logging

    # Apply configuration overrides before starting if provided
    if config_path is not None:
        global global_config
        global_config = load_config(config_path)

    # Set up logging
    logger = setup_logging(log_level=log_level, json_log_file=json_log_file)
    logger.info(f"Starting blur API server on {host}:{port}")

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower(), reload=reload)


def main():
    parser = argparse.ArgumentParser(description="Start the blur API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument("--json-log", type=Path, help="Enable JSON logging to specified file")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a TOML configuration file to load before starting the server",
    )

    args = parser.parse_args()
    start_server(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        json_log_file=args.json_log,
        reload=args.reload,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
