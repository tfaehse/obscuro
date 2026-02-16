# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Python Development
```bash
# Install dependencies (Python 3.12+)
uv sync
uv sync --extra gpu  # For CUDA support

# Run tests
uv run pytest                          # All tests
uv run pytest tests/test_foo.py         # Single file
uv run pytest tests/test_foo.py -k test_bar  # Specific test
uv run pytest -n auto                  # Parallel with pytest-xdist

# Linting and type checking
uv run ruff check                      # Linter
uv run ruff format                     # Formatter
uv run ty check.                       # Type checker
uv run bandit check .                  # Security linter (with config in pyproject.toml)

# Git hooks (optional)
pre-commit install                     # Install git hooks for auto-formatting

# Run services
uv run blur-api                        # Start FastAPI backend (http://localhost:8000)
uv run blur-cli image input.jpg        # Process single image
uv run blur-cli video input.mp4         # Process video
```

### Frontend Development
```bash
cd src/frontend
npm install
npm run dev                    # Start Electron in dev mode
npm run build:ts              # Compile TypeScript
npm run build                 # Build desktop app package
```

## Architecture Overview

The anonymizer follows a three-stage pipeline: **detect → track → blur**. All state flows through Polars DataFrames and `AnonymizerConfig`.

### Core Pipeline (`src/anonymizer/core.py`)

The `Anonymizer` class orchestrates:
1. **Detection**: ONNX inference with SAHI tiling support (`Detector` → `SahiDetector`)
2. **Tracking**: Multi-object tracking via tracker factory (`TrackerFactory.get()`)
3. **Blurring**: Apply blur effects to detected/tracked regions (`Blurrer`)

### Configuration System (`src/anonymizer/config.py`)

- `AnonymizerConfig`: Pydantic-based configuration with baked-in defaults
- `ConfigLayers`: Helper for composing configuration from ordered override layers (e.g., base → TOML → CLI → request)
- Override syntax: Supports nested dicts, dotted keys (`detection.confidence_threshold`), or double-underscore keys (`detection__confidence_threshold`)
- `load_config()`: Load from TOML file with optional overrides

### Detection (`src/anonymizer/detection/`)

- `Detector`: Factory returning `SahiDetector`
- `SahiDetector`: SAHI (Slicing Aided Hyper Inference) wrapper for ONNX models
- Detection results: Polars DataFrame with columns for bounding boxes, confidence scores, and segmentation masks
- `model_metadata.py`: Loads detector class names from bundled `model_metadata.json` or fallback files

### Tracking (`src/anonymizer/tracking/`)

Tracker implementations via factory pattern:
- `DummyTracker`: No tracking (one-off detections)
- `ByteTrackTracker`: Low-score pool + Kalman filter
- `BotSortTracker`: Camera motion compensation + Kalman
- `FusedTracker`: Combined distance + shape + embedding similarity
- `HybridSOTTracker`: Single-object tracker bridging detector gaps (visual tracker backend)
- `OCSortTracker`: Observation-centric tracking for non-linear motion (IoU-based association, handles complex trajectories)

Key tracking concepts:
- `TrackerParams`: Normalized hyperparameters shared across implementations
- `DEFAULT_TRACKER_PARAMS`: Tracker-specific defaults (distance_gate, confirm_after_N, max_misses_M, etc.)
- `offline_linker`: Post-hoc tracklet linking via Hungarian/LAP algorithm
- `ActiveTrack` uses `slots=True` - cannot add arbitrary attributes. Use separate dict keyed by `track_id` for tracker-specific state.

### Interfaces

**CLI** (`src/blur_cli/cli.py`):
- Subcommands: `image`, `video`, `config`, `models`
- CLI arguments map to config overrides via `get_config_for_args()`
- Supports TOML config files with `--config` flag

**FastAPI** (`src/blur_api/serve.py`):
- Async endpoints with job-based video processing
- GPU/CPU execution provider selection with model caching (`get_or_create_gpu_anonymizer`, `get_or_create_cpu_anonymizer`)
- Server-Sent Events for progress streaming (`EventSourceResponse`)
- Configuration is layered: global base config + per-request overrides via `build_effective_config()`
- Session directories: created lazily via `get_session_temp_dir()` under `get_temp_dir() / "obscuro_jobs"`
- Module-level variables (TEMP_ROOT, SESSION_TEMP_DIR) cause issues with uvicorn --reload

**Electron Frontend** (`src/frontend/app/renderer`):
- TypeScript + vanilla DOM helpers (no React framework)
- Live video preview with frame-by-frame anonymization
- Config controller handles parameter updates and sends to backend
- Blur preview generates on: video pause, video load, next/prev frame navigation
- AbortController cancels pending preview requests during rapid frame navigation
- Run `npm run build:ts` after TypeScript changes to verify compilation

### Data Flow

1. Detections: Polars DataFrame with columns `frame`, `tlwh`, `confidence`, `class_name`, and optional `mask` (default classes: `["plate", "head"]`)
2. Tracking: `Tracker.track(detections)` returns tracked results with `track_id` column
3. Offline linker (optional): Merges broken tracklets post-processing via `link_tracklets()`
4. Blur: `Blurrer.blur_video()` applies blur to tracked regions using bounding boxes or masks

### Important Constraints

- **SAHI tiling**: Controlled via `config.detection.use_sahi` (CLI: `--use-sahi/--no-use-sahi`)
- **SAHI single-pass mode**: `config.detection.single_pass` uses single tile with no overlap (faster, may miss small objects)
- **Segmentation masks**: Disabled with `config.detection.disable_masks` (CLI: `--disable-masks`)
- **Blur types**: `gaussian`, `pixelate`, `blackout`, `debug` (debug mode draws colored overlays)
- **Tracker params** are validated through `TrackerParams` Pydantic model

### Logging and Progress

- Progress callback: `Callable[[int, str, str], None]` → `(percentage, stage, message)`
- Throttled via `throttle_progress_callback()` (2-second intervals, 5% steps)
- Cancellation: `threading.Event()` checked throughout pipeline
- Logger instances: `obscuro`, `obscuro.detection`, `obscuro.tracking`, `obscuro.api`

### Model Management

- Models stored in `get_detection_models_dir()` (platform-specific)
- Default model: `DEFAULT_MODEL_NAME` from `paths.py`
- Immutable models (bundled): `IMMUTABLE_MODEL_NAMES` - cannot be deleted via API
- Model-specific batch constraints: Models ending in `_b1` enforce batch size 1 (CoreML export)
- `model_requires_static_batch()` and `enforce_model_batch_constraints()` handle this

### Paths and Resources

- `src/anonymizer/paths.py`: Platform-aware model directories, model bundling
- `models/detection/`: ONNX detection models
- `models/tracking/`: Optional tracking backends (e.g., TrackerNano weights - not bundled)
- `scripts/`: Development utilities for analysis and benchmarking (analyze_trace.py, benchmark_tracking.py, export_yolo11n_seg.py)

### Test Coverage

- Tests located in `tests/`
- Coverage target: 80% (`--cov-fail-under=80` in pytest config)
- `tests/anonymizer/test_*`: Core pipeline tests
- `tests/blur_api/test_*`: API endpoint tests
- `tests/blur_gui/test_*`: CLI tests

### Adding a New Tracker

1. Create tracker class in `src/anonymizer/tracking/<name>.py` extending `BaseTracker`
2. Add to `TRACKER_FACTORY` in `src/anonymizer/tracking/__init__.py`
3. Add `TrackerType.<NAME>` enum value in `src/anonymizer/config.py`
4. Add default params to `DEFAULT_TRACKER_PARAMS` in `src/anonymizer/config.py`
5. Update `TrackerType` in `src/frontend/app/renderer/types/api.ts`
6. Add tests in `tests/anonymizer/tracking/test_trackers.py`
