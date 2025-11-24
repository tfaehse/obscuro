# Python Library API

This guide covers using Obscuro as a Python library for programmatic video anonymization.

## Installation

```bash
# Install with uv
uv sync

# Or with pip
pip install -e .
```

## Quick Start

```python
from pathlib import Path
from anonymizer import Anonymizer, AnonymizerConfig

# Create configuration with defaults
config = AnonymizerConfig()

# Initialize anonymizer
anonymizer = Anonymizer(config=config)

# Process a video
anonymizer.blur_video(Path("input.mp4"), Path("output.mp4"))
```

## Core Classes

### `AnonymizerConfig`

Configuration class using Pydantic models for validation.

```python
from anonymizer import AnonymizerConfig
from anonymizer.config import BlurType, TrackerType

# Create with defaults
config = AnonymizerConfig()

# Create with custom settings
config = AnonymizerConfig(
    model={"name": "1280_nano"},
    blur={"type": BlurType.PIXELATE, "strength": 20},
    detection={
        "confidence_threshold": 0.3,
        "low_score_threshold": 0.1,
        "batch_size": 8,
        "use_sahi": True,
        "inference_size": 2560
    },
    tracking={
        "type": TrackerType.BOTSORT,
        "use_offline_linker": True
    },
    video={"codec": "h264", "quality": 23},
    debug=False,
    log_level="INFO"
)
```

#### Loading from TOML

```python
from pathlib import Path
from anonymizer import load_config

# Load configuration from file
config = load_config(Path("config.toml"))
```

#### Configuration Layers

For advanced use cases with runtime overrides:

```python
from anonymizer.config import ConfigLayers, set_config

# Create base config
base_config = AnonymizerConfig()

# Create layers
layers = ConfigLayers(base_config)

# Add override layers
layers.set_layer("user", {
    "blur": {"type": "pixelate", "strength": 25}
})

layers.set_layer("runtime", {
    "detection": {"batch_size": 16}
})

# Resolve final configuration
final_config = layers.resolve()

# Set as global config
set_config(final_config)
```

### `Anonymizer`

Main class for video and image processing.

```python
from anonymizer import Anonymizer

anonymizer = Anonymizer(
    config=config,
    progress_callback=None,
    cancel_event=None,
    execution_providers=None
)
```

#### Parameters

- **`config`** (`AnonymizerConfig`) - Configuration object
- **`progress_callback`** (`Callable[[int, str, str], None]`, optional) - Progress callback function
- **`cancel_event`** (`threading.Event`, optional) - Event for cancellation support
- **`execution_providers`** (`list[str]`, optional) - ONNX Runtime execution providers
  - `None` (default) - Auto-detect GPU, fallback to CPU
  - `["CUDAExecutionProvider", "CPUExecutionProvider"]` - Force GPU with CPU fallback
  - `["CPUExecutionProvider"]` - Force CPU only

#### Methods

##### `blur_video(input_path, output_path) -> int`

Process a video file with blur effects.

```python
from pathlib import Path

result = anonymizer.blur_video(
    input_path=Path("input.mp4"),
    output_path=Path("output.mp4")
)

# Returns 0 on success, non-zero on error
if result == 0:
    print("Success!")
```

##### `blur_image_file(input_path, output_path)`

Process an image file with blur effects.

```python
anonymizer.blur_image_file(
    input_path=Path("input.jpg"),
    output_path=Path("output.jpg")
)
```

##### `blur_image_array(image_array) -> np.ndarray`

Process a numpy array (OpenCV image) directly.

```python
import cv2

# Read image
image = cv2.imread("input.jpg")

# Process
blurred = anonymizer.blur_image_array(image)

# Save result
cv2.imwrite("output.jpg", blurred)
```

##### `set_runtime_hooks(cancel_event, progress_callback)`

Update runtime hooks (for reusing anonymizer instances).

```python
import threading

cancel_event = threading.Event()

def progress(percent, stage, message):
    print(f"{percent}% - {stage}: {message}")

anonymizer.set_runtime_hooks(
    cancel_event=cancel_event,
    progress_callback=progress
)
```

## Progress Tracking

Implement a progress callback to monitor processing:

```python
def progress_callback(percent: int, stage: str, message: str):
    """
    Args:
        percent: Progress percentage (0-100)
        stage: Current stage (e.g., "Detection", "Tracking", "Blurring")
        message: Detailed message about current operation
    """
    print(f"[{stage}] {percent}% - {message}")

anonymizer = Anonymizer(
    config=config,
    progress_callback=progress_callback
)

anonymizer.blur_video(Path("input.mp4"), Path("output.mp4"))
```

### Example Output

```
[Detection] 10% - Processing frame 50/500
[Detection] 25% - Processing frame 125/500
[Detection] 50% - Processing frame 250/500
[Tracking] 60% - Associating detections across frames
[Tracking] 65% - Building tracklets
[Blurring] 70% - Applying blur effects
[Blurring] 85% - Processing frame 425/500
[Blurring] 100% - Complete
```

## Cancellation Support

Implement cancellation with threading.Event:

```python
import threading
from anonymizer.cancellation import CancellationException

cancel_event = threading.Event()

anonymizer = Anonymizer(
    config=config,
    cancel_event=cancel_event
)

# Start processing in background thread
def process():
    try:
        anonymizer.blur_video(Path("input.mp4"), Path("output.mp4"))
        print("Processing complete")
    except CancellationException:
        print("Processing cancelled")

thread = threading.Thread(target=process)
thread.start()

# Cancel after some time
import time
time.sleep(5)
cancel_event.set()  # Trigger cancellation

thread.join()
```

## Advanced Configuration

### Custom Model Path

```python
from pathlib import Path

config = AnonymizerConfig(
    model={"file": Path("/path/to/custom_model.onnx")}
)
```

### SAHI Configuration

```python
config = AnonymizerConfig(
    detection={
        "use_sahi": True,
        "inference_size": 2560,
        "sahi_overlap_ratio": 0.25
    }
)
```

### Tracker Parameters

```python
config = AnonymizerConfig(
    tracking={
        "type": TrackerType.BOTSORT,
        "use_offline_linker": True,
        "params": {
            "distance_gate": 0.1,
            "confirm_after_N": 3,
            "max_misses_M": 15,
            "cam_motion_comp": True
        }
    }
)
```

### GPU/CPU Selection

```python
# Force CPU only
anonymizer = Anonymizer(
    config=config,
    execution_providers=["CPUExecutionProvider"]
)

# Force GPU (CUDA)
anonymizer = Anonymizer(
    config=config,
    execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Auto-detect (default)
anonymizer = Anonymizer(config=config)
```

## Batch Processing

Process multiple videos with shared configuration:

```python
from pathlib import Path

config = AnonymizerConfig()
anonymizer = Anonymizer(config=config)

videos = [
    ("video1.mp4", "output1.mp4"),
    ("video2.mp4", "output2.mp4"),
    ("video3.mp4", "output3.mp4"),
]

for input_file, output_file in videos:
    print(f"Processing {input_file}...")
    result = anonymizer.blur_video(Path(input_file), Path(output_file))
    if result == 0:
        print(f"✓ {output_file}")
    else:
        print(f"✗ {input_file} failed")
```

## Custom Detection Model

### Load Custom ONNX Model

```python
from pathlib import Path

config = AnonymizerConfig(
    model={"file": Path("/path/to/custom_model.onnx")},
    detection={
        "confidence_threshold": 0.4,
        "low_score_threshold": 0.1
    }
)

anonymizer = Anonymizer(config=config)
```

## Error Handling

```python
from pathlib import Path
from anonymizer.cancellation import CancellationException

try:
    anonymizer.blur_video(Path("input.mp4"), Path("output.mp4"))
except CancellationException:
    print("Processing was cancelled")
except FileNotFoundError:
    print("Input file not found")
except Exception as e:
    print(f"Error: {e}")
```

## Complete Examples

### Example 1: Simple Video Processing

```python
from pathlib import Path
from anonymizer import Anonymizer, AnonymizerConfig

def process_video(input_path: Path, output_path: Path):
    """Process a video with default settings."""
    config = AnonymizerConfig()
    anonymizer = Anonymizer(config=config)

    result = anonymizer.blur_video(input_path, output_path)
    return result == 0

# Usage
success = process_video(Path("dashcam.mp4"), Path("blurred.mp4"))
print(f"Success: {success}")
```

### Example 2: Custom Configuration with Progress

```python
from pathlib import Path
from anonymizer import Anonymizer, AnonymizerConfig
from anonymizer.config import BlurType, TrackerType

def process_with_progress(input_path: Path, output_path: Path):
    """Process video with custom settings and progress tracking."""

    # Progress callback
    def on_progress(percent: int, stage: str, message: str):
        print(f"\r{stage}: {percent}%", end="", flush=True)

    # Configuration
    config = AnonymizerConfig(
        blur={"type": BlurType.PIXELATE, "strength": 20},
        detection={
            "confidence_threshold": 0.3,
            "low_score_threshold": 0.1,
            "batch_size": 8
        },
        tracking={"type": TrackerType.BOTSORT},
        video={"codec": "h264", "quality": 23}
    )

    # Process
    anonymizer = Anonymizer(config=config, progress_callback=on_progress)
    result = anonymizer.blur_video(input_path, output_path)

    print()  # New line after progress
    return result == 0

# Usage
success = process_with_progress(Path("input.mp4"), Path("output.mp4"))
```

### Example 3: Batch Processing with Cancellation

```python
import threading
from pathlib import Path
from anonymizer import Anonymizer, AnonymizerConfig
from anonymizer.cancellation import CancellationException

class BatchProcessor:
    def __init__(self):
        self.config = AnonymizerConfig()
        self.cancel_event = threading.Event()
        self.current_file = None

    def process_batch(self, file_pairs: list[tuple[Path, Path]]):
        """Process multiple videos with cancellation support."""
        results = []

        for input_path, output_path in file_pairs:
            if self.cancel_event.is_set():
                print(f"Batch cancelled at {input_path}")
                break

            self.current_file = input_path
            print(f"Processing {input_path}...")

            try:
                anonymizer = Anonymizer(
                    config=self.config,
                    cancel_event=self.cancel_event
                )
                result = anonymizer.blur_video(input_path, output_path)
                results.append((input_path, result == 0))
            except CancellationException:
                print(f"Cancelled: {input_path}")
                results.append((input_path, False))
                break
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
                results.append((input_path, False))

        return results

    def cancel(self):
        """Cancel current processing."""
        self.cancel_event.set()

# Usage
processor = BatchProcessor()

videos = [
    (Path("video1.mp4"), Path("output1.mp4")),
    (Path("video2.mp4"), Path("output2.mp4")),
    (Path("video3.mp4"), Path("output3.mp4")),
]

# Process in background thread
thread = threading.Thread(target=lambda: processor.process_batch(videos))
thread.start()

# Cancel if needed
# processor.cancel()

thread.join()
```

### Example 4: Image Processing

```python
from pathlib import Path
import cv2
from anonymizer import Anonymizer, AnonymizerConfig
from anonymizer.config import BlurType

def process_image_file(input_path: Path, output_path: Path):
    """Process an image file."""
    config = AnonymizerConfig(
        blur={"type": BlurType.GAUSSIAN, "strength": 15}
    )

    anonymizer = Anonymizer(config=config)
    anonymizer.blur_image_file(input_path, output_path)

def process_image_array(image_path: Path):
    """Process image as numpy array."""
    config = AnonymizerConfig()
    anonymizer = Anonymizer(config=config)

    # Load image
    image = cv2.imread(str(image_path))

    # Process
    blurred = anonymizer.blur_image_array(image)

    # Save or display
    cv2.imwrite("output.jpg", blurred)
    return blurred

# Usage
process_image_file(Path("input.jpg"), Path("output.jpg"))
```

### Example 5: High-Resolution Video with SAHI

```python
from pathlib import Path
from anonymizer import Anonymizer, AnonymizerConfig
from anonymizer.config import TrackerType

def process_4k_video(input_path: Path, output_path: Path):
    """Process 4K video with SAHI tiled inference."""

    config = AnonymizerConfig(
        detection={
            "use_sahi": True,
            "inference_size": 3840,
            "sahi_overlap_ratio": 0.25,
            "batch_size": 4,
            "confidence_threshold": 0.3,
            "low_score_threshold": 0.1
        },
        tracking={
            "type": TrackerType.BOTSORT,
            "params": {
                "max_misses_M": 15,
                "offline_linker_max_misses": 60
            }
        },
        video={
            "codec": "hevc",
            "quality": 22
        }
    )

    def progress(percent, stage, message):
        print(f"[{stage}] {percent}%: {message}")

    anonymizer = Anonymizer(config=config, progress_callback=progress)
    return anonymizer.blur_video(input_path, output_path) == 0

# Usage
success = process_4k_video(Path("4k_input.mp4"), Path("4k_output.mp4"))
```

## Helper Functions

### Create Configuration Template

```python
from anonymizer import create_config_template
from pathlib import Path

# Create a TOML config template
create_config_template(Path("my_config.toml"))
```

### Load and Apply Configuration

```python
from anonymizer import load_config, set_config

# Load config from file
config = load_config(Path("my_config.toml"))

# Set as global config (affects all subsequent Anonymizer instances)
set_config(config)

# Or pass directly to Anonymizer
anonymizer = Anonymizer(config=config)
```

## Logging

Configure logging for the library:

```python
from blur_gui.logging_setup import setup_logging

# Setup logging
logger = setup_logging(
    log_level="DEBUG",
    json_log_file=Path("processing.json"),
    enable_colors=True
)

# Now process videos
anonymizer = Anonymizer(config=config)
anonymizer.blur_video(Path("input.mp4"), Path("output.mp4"))
```

## Model Management

### Ensure Model Exists

```python
from anonymizer.paths import ensure_default_model_present, get_detection_models_dir

# Ensure default detection model is present
models_dir = get_detection_models_dir()
ensure_default_model_present(models_dir)
```

### List Available Models

```python
from pathlib import Path
from anonymizer.paths import get_detection_models_dir

models_dir = get_detection_models_dir()
models = list(models_dir.glob("*.onnx"))

for model in models:
    print(f"Model: {model.stem}, Size: {model.stat().st_size / 1024 / 1024:.1f} MB")
```

## Performance Tips

1. **Reuse Anonymizer instances** - Model loading is expensive, reuse for multiple videos
2. **Increase batch_size** - For GPU processing, try 8-16 for better throughput
3. **Use appropriate tracker** - `dummy` is fastest, `botsort` is best quality
4. **Adjust inference_size** - Lower values process faster but may miss small objects
5. **Disable SAHI** - Unless you need it for high-res or distant objects
6. **Force CPU** - Use `execution_providers=["CPUExecutionProvider"]` if GPU is unavailable

## Troubleshooting

### Out of Memory

```python
config = AnonymizerConfig(
    detection={
        "batch_size": 2,  # Reduce batch size
        "inference_size": 1280  # Lower resolution
    }
)
```

### Slow Processing

```python
config = AnonymizerConfig(
    tracking={"type": TrackerType.DUMMY},  # Skip tracking
    detection={"batch_size": 16}  # Increase batch size
)
```

### Missing Detections

```python
config = AnonymizerConfig(
    detection={
        "confidence_threshold": 0.2,  # Lower threshold
        "low_score_threshold": 0.1,
        "use_sahi": True  # Enable tiled inference
    }
)
```

## API Reference Summary

### Main Classes
- `Anonymizer` - Main processing class
- `AnonymizerConfig` - Configuration management
- `ConfigLayers` - Multi-layer configuration system

### Enums
- `BlurType` - Blur effect types (GAUSSIAN, PIXELATE, BLACKOUT, BLACK, DEBUG)
- `TrackerType` - Tracker algorithms (DUMMY, BYTETRACK, BOTSORT, HYBRID_SOT)

### Helper Functions
- `load_config(path)` - Load configuration from TOML
- `create_config_template(path)` - Generate config template
- `set_config(config)` - Set global configuration
- `ensure_default_model_present(dir)` - Ensure default model exists
- `get_models_dir()` - Get the root models directory (`detection/` + `tracking/`)
- `get_detection_models_dir()` / `get_tracking_models_dir()` - Get the specific subdirectories

### Exceptions
- `CancellationException` - Raised when processing is cancelled
