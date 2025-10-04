# Copilot Instructions for blur-gui

## Project Overview
This is a **dashcam video anonymization system** that detects, tracks, and blurs faces and license plates in videos. The architecture follows a **three-stage pipeline**: Detection → Tracking → Blurring.

**Goal**: Package as a cross-platform desktop application with CLI, API, and GUI interfaces.

## Core Architecture

### Multi-Interface Design
- **CLI**: `src/blur_gui/cli.py` - Command-line interface for batch processing
- **API**: `src/blur_api/serve.py` - FastAPI server for programmatic access
- **Frontend**: `src/frontend/index.html` - HTML5 web interface with real-time preview
- **Core**: `src/anonymizer/` - Shared processing engine

### Processing Pipeline (src/anonymizer/core.py)
```python
# Three-stage pipeline with progress reporting
def blur_video(self, input_path, output_path):
    detections = self._detect(input_path)      # Neural network inference
    tracks = self.tracker.track(detections)    # Object tracking across frames
    self.blurrer.blur_video(input_path, tracks, output_path)  # Apply blur masks
```

### Key Components
- **Detection**: `detection.py` - ONNX neural network inference with batch processing
- **Tracking**: `tracking/` - Multiple algorithms (MinCostFlow, BotSort, Dummy)
- **Blurring**: `blurring.py` - PyAV-based video processing with multiple blur types
- **Config**: `config.py` - Pydantic-based configuration with environment variable support

## Development Standards

### Packaging & Compatibility
- **Target**: Cross-platform desktop application (Windows, macOS, Linux)
- **Technology stack**: Python + Electron frontend + PyInstaller for distribution
- **Python**: 3.12+ only - no backwards compatibility needed
- **Version**: v1 development - no backwards compatibility requirements

### Code Patterns
- **Path handling**: Use `pathlib.Path` exclusively, never `os.path`
- **DataFrames**: Polars for all tabular data - only native types, no Python objects in schemas
- **Type hints**: Full typing
- **Imports**: Relative imports within packages, absolute for cross-package

### Build System
```bash
# Use nox for all build/test operations (replaces make/shell scripts)
nox -s test              # Run all tests with coverage
nox -s build             # Build Python + frontend
nox -s lint format       # Code quality checks
nox -s test-fast         # Quick tests without coverage
```

### Running the System
```bash
# CLI usage (single-shot processing)
python -m blur_gui.cli video input.mp4 --blur-type pixelate

# API server (for frontend/integration)
python -m blur_api.serve --host 0.0.0.0 --port 8000
```

## Critical Patterns

### Configuration System
- **Layered config**: Environment variables > CLI args > config files > defaults
- **Live updates**: API supports runtime config overrides
- **TOML files**: `config/default.toml`, `config/dev.toml`, `config/production.toml`

### Cancellation & Progress
- **Thread-safe cancellation**: All components support cancellation via threading.Event
- **Structured progress**: Progress callbacks with stage names and percentages
- **Graceful cleanup**: Temporary files and resources cleaned up on cancellation

### Video Processing
- **PyAV backend**: `src/anonymizer/io/video.py` - Efficient frame-by-frame processing
- **Memory management**: Streaming processing for large videos, no frame caching
- **Audio preservation**: Pass-through audio streams without re-encoding

### Error Handling
- **Custom exceptions**: `CancellationException` for clean cancellation handling
- **Structured logging**: JSON logging support with `blur_gui.logging_setup`
- **Resource cleanup**: Use finally blocks and context managers

## Testing Strategy

### Test Categories (tests/)
- **Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`
- **Fixtures**: `conftest.py` provides shared test data and mocks
- **Coverage**: Use `nox -s coverage` for detailed reports

### Code Quality
- **Ruff**: Both linting and formatting (replaces black + flake8)
- **Pre-commit hooks**: Enforce quality before commits
- **Security**: Bandit for security linting

## Integration Points

### Frontend ↔ API
- **File upload**: FormData with video file + JSON config
- **Progress tracking**: Server-Sent Events (SSE) for real-time updates
- **Real-time preview**: Single frame processing for UI preview

### Model Loading
- **ONNX models**: Stored in `models/` directory, auto-discovered
- **Lazy loading**: Models loaded on first use
- **Thread safety**: Protected model access in multi-threaded API

### Data Flow
- **Polars DataFrames**: Primary data structure throughout pipeline
- **Coordinate system**: Relative coordinates (0.0-1.0) for resolution independence
- **Native types only**: DataFrames contain only built-in Python types, no objects

## Key Files for New Contributors
- `src/anonymizer/core.py` - Main orchestration logic
- `src/blur_api/serve.py` - API endpoints and job management
- `tests/conftest.py` - Test fixtures and shared utilities
- `noxfile.py` - Build system and development workflows
- `src/anonymizer/config.py` - Configuration schema and validation
