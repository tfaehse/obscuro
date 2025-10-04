# Obscuro Documentation

Welcome to the comprehensive documentation for **Obscuro**, a dashcam video anonymization system for detecting, tracking, and blurring faces and license plates in videos.

## Quick Links

- [CLI Reference](cli-reference.md) - Command-line interface usage
- [API Reference](api-reference.md) - FastAPI REST endpoints
- [Configuration Guide](configuration.md) - All configuration options explained
- [Python Library](python-api.md) - Using the anonymizer as a Python library

## Overview

Obscuro provides three interfaces for video anonymization:

1. **Command-Line Interface (CLI)** - For batch processing and automation
2. **REST API** - For integration with other applications
3. **Desktop GUI** - For interactive video processing

All interfaces share the same core processing pipeline:

```
Detection → Tracking → Blurring
```

### Processing Pipeline

1. **Detection**: Neural network-based detection of faces and license plates
2. **Tracking**: Multi-object tracking to associate detections across frames
3. **Blurring**: Apply blur effects (Gaussian, pixelate, blackout, etc.)

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install from source

```bash
# Clone the repository
git clone https://github.com/tfaehse/obscuro.git
cd obscuro

# Install with uv
uv sync

# Or install with pip
pip install -e .
```

## Quick Start Examples

### CLI

```bash
# Blur a video with default settings
blur-cli video input.mp4

# Blur an image with custom settings
blur-cli image input.jpg --blur-type pixelate --blur-strength 20

# Generate a configuration template
blur-cli config -o my_config.toml
```

### Python Library

```python
from pathlib import Path
from anonymizer import Anonymizer, AnonymizerConfig

# Create configuration
config = AnonymizerConfig()

# Initialize anonymizer
anonymizer = Anonymizer(config=config)

# Process video
anonymizer.blur_video(Path("input.mp4"), Path("output.mp4"))
```

### API Server

```bash
# Start the API server
blur-api --host 0.0.0.0 --port 8000

# Or use uvicorn directly
uvicorn blur_api.serve:app --reload
```

## Key Features

- **Multiple blur types**: Gaussian, pixelate, blackout, black boxes
- **Advanced tracking**: ByteTrack, BotSort, Hybrid SOT, or simple frame-by-frame
- **SAHI support**: Tiled inference for high-resolution videos
- **GPU acceleration**: Automatic GPU detection with CPU fallback
- **Progress tracking**: Real-time progress callbacks and cancellation support
- **Flexible configuration**: TOML files, environment variables, CLI arguments
- **Model management**: Upload, download, and manage ONNX models

## Next Steps

- Read the [CLI Reference](cli-reference.md) for command-line usage
- Explore the [Configuration Guide](configuration.md) for detailed options
- Check the [API Reference](api-reference.md) for REST endpoints
- Learn about the [Python Library](python-api.md) for programmatic usage
