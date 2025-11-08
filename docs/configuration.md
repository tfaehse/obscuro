# Configuration Guide

This guide covers all configuration options available in Obscuro. Configuration can be provided via TOML files or command-line/API overrides.

## Configuration Priority

Settings are loaded in the following order (later sources override earlier):

1. **Built-in defaults**
2. **TOML configuration file** (via `--config` flag or loaded programmatically)
3. **Command-line arguments** or **API request parameters**

## Configuration Structure

The configuration is organized into nested sections:

```toml
# Global settings
debug = false
log_level = "INFO"

[model]
name = "1280_nano"
# file = "/path/to/custom_model.onnx"  # Alternative to name

[blur]
type = "gaussian"
strength = 10

[detection]
plate_threshold = 0.5
face_threshold = 0.5
batch_size = 4
use_sahi = true
inference_size = 1920
sahi_overlap_ratio = 0.2

[tracking]
type = "bytetrack"
use_offline_linker = true

[tracking.params]
distance_gate = 0.05
confirm_after_N = 2
max_misses_M = 10
# ... additional parameters

[video]
codec = "h264"
quality = 23  # Optional, lower = better quality
```

## Global Settings

### `debug`
**Type:** Boolean
**Default:** `false`

Enable debug mode for verbose output and additional diagnostics.

**TOML:**
```toml
debug = true
```

**CLI:**
```bash
blur-cli --config-debug video input.mp4
```

### `log_level`
**Type:** String
**Default:** `"INFO"`
**Choices:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

Set the logging verbosity level.

**TOML:**
```toml
log_level = "DEBUG"
```

**CLI:**
```bash
blur-cli --log-level DEBUG video input.mp4
```

---

## Model Configuration

### `model.name`
**Type:** String
**Default:** `"1280_nano"`

Name of the ONNX model file (without .onnx extension). The model must exist in the models directory.

**TOML:**
```toml
[model]
name = "640_nano"
```

**CLI:**
```bash
blur-cli image input.jpg --model 640_nano
```

### `model.file`
**Type:** Path
**Default:** `null`

Full path to an ONNX model file. If specified, overrides `model.name`.

**TOML:**
```toml
[model]
file = "/path/to/custom_model.onnx"
```

**CLI:**
```bash
blur-cli image input.jpg --model /path/to/custom_model.onnx
```

**Note:** Use either `name` or `file`, not both.

---

## Blur Configuration

### `blur.type`
**Type:** String
**Default:** `"gaussian"`
**Choices:** `gaussian`, `pixelate`, `blackout`, `debug`

Type of blur effect to apply:

- **`gaussian`** - Smooth Gaussian blur (most natural looking)
- **`pixelate`** - Pixelated/mosaic effect
- **`blackout`** - Solid black boxes
- **`debug`** - Shows bounding boxes without blurring (for debugging)

**TOML:**
```toml
[blur]
type = "pixelate"
```

**CLI:**
```bash
blur-cli image input.jpg --blur-type pixelate
```

**API:**
```json
{
  "blur": {
    "type": "pixelate"
  }
}
```

### `blur.strength`
**Type:** Integer
**Default:** `10`
**Range:** `1-100`

Blur strength/intensity. Higher values produce stronger blur effects.

**TOML:**
```toml
[blur]
strength = 25
```

**CLI:**
```bash
blur-cli image input.jpg --blur-strength 25
```

**API:**
```json
{
  "blur": {
    "strength": 25
  }
}
```

---

## Detection Configuration

### `detection.plate_threshold`
**Type:** Float
**Default:** `0.5`
**Range:** `0.0-1.0`

Confidence threshold for license plate detection. Lower values detect more plates but may include false positives.

**TOML:**
```toml
[detection]
plate_threshold = 0.3
```

**CLI:**
```bash
blur-cli image input.jpg --plate-threshold 0.3
```

**API:**
```json
{
  "detection": {
    "plate_threshold": 0.3
  }
}
```

### `detection.face_threshold`
**Type:** Float
**Default:** `0.5`
**Range:** `0.0-1.0`

Confidence threshold for face detection.

**TOML:**
```toml
[detection]
face_threshold = 0.6
```

**CLI:**
```bash
blur-cli image input.jpg --face-threshold 0.6
```

### `detection.batch_size`
**Type:** Integer
**Default:** `4`
**Range:** `1-256`

Number of frames/images processed per detector forward pass. Higher values can improve GPU throughput but require more memory.

**Recommendations:**
- GPU with 4GB VRAM: `batch_size = 4-8`
- GPU with 8GB+ VRAM: `batch_size = 8-16`
- CPU only: `batch_size = 1-2`

**TOML:**
```toml
[detection]
batch_size = 8
```

**CLI:**
```bash
blur-cli video input.mp4 --batch-size 8
```

> Models whose names end with `_b1` are optimized for Core ML and only support `batch_size = 1`.
> The CLI/API/GUI automatically lock the batch slider to `1` whenever such a model is selected.
> Using these models on non-Apple Silicon platforms largely makes no sense.

### `detection.use_sahi`
**Type:** Boolean
**Default:** `false`

Enable SAHI (Slicing Aided Hyper Inference) for tiled inference. Useful for:
- High-resolution videos (4K+)
- Detecting small/distant objects
- Videos where objects appear at various scales

**TOML:**
```toml
[detection]
use_sahi = true
```

**CLI:**
```bash
blur-cli video input.mp4 --use-sahi
```

**API:**
```json
{
  "detection": {
    "use_sahi": true
  }
}
```

### `detection.inference_size`
**Type:** Integer
**Default:** `1920`
**Range:** `256-8192`

Longest image edge (in pixels) used for detection inference. Images larger than this are downscaled before detection, then detections are scaled back up.

**Recommendations:**
- `1280_nano` model: `inference_size = 1280`
- `640_nano` model: `inference_size = 640`
- 4K video: `inference_size = 2560-3840`

**TOML:**
```toml
[detection]
inference_size = 2560
```

**CLI:**
```bash
blur-cli video 4k_video.mp4 --inference-size 2560
```

### `detection.sahi_overlap_ratio`
**Type:** Float
**Default:** `0.2`
**Range:** `0.0-0.99`

Overlap ratio between SAHI tiles. Higher values improve detection at tile boundaries but increase computation time.

**TOML:**
```toml
[detection]
sahi_overlap_ratio = 0.3
```

**CLI:**
```bash
blur-cli video input.mp4 --use-sahi --sahi-overlap 0.3
```

---

## Tracking Configuration

### `tracking.type`
**Type:** String
**Default:** `"bytetrack"`
**Choices:** `dummy`, `bytetrack`, `botsort`, `hybrid_sot`

Multi-object tracker algorithm:

- **`dummy`** - No tracking, frame-by-frame detection only (fastest)
- **`bytetrack`** - Fast ByteTrack algorithm (good balance of speed/quality)
- **`botsort`** - BoT-SORT with camera motion compensation (best quality)
- **`hybrid_sot`** - Hybrid single-object tracking (experimental)

**TOML:**
```toml
[tracking]
type = "botsort"
```

**CLI:**
```bash
blur-cli video input.mp4 --tracker botsort
```

**API:**
```json
{
  "tracking": {
    "type": "botsort"
  }
}
```

### `tracking.use_offline_linker`
**Type:** Boolean
**Default:** `true`

Enable offline tracklet linking (post-processing pass to reconnect broken tracks). Recommended to keep enabled for better tracking quality.

**TOML:**
```toml
[tracking]
use_offline_linker = true
```

**CLI:**
```bash
blur-cli video input.mp4 --no-offline-linker  # Disable
```

---

## Tracker Parameters

Each tracker type has specific parameters that can be tuned. Parameters are stored in `tracking.params` and have sensible defaults per tracker.

### Common Parameters

#### `distance_gate`
**Type:** Float
**Default:** Varies by tracker
**Range:** `0.0-1.0`

Maximum distance threshold for associating detections to tracks (in normalized coordinates).

**TOML:**
```toml
[tracking.params]
distance_gate = 0.1
```

**CLI:**
```bash
blur-cli video input.mp4 --tracker-params '{"distance_gate":0.1}'
```

#### `confirm_after_N`
**Type:** Integer
**Default:** Varies by tracker
**Range:** `1-10`

Number of consecutive detections required before a track is confirmed.

**TOML:**
```toml
[tracking.params]
confirm_after_N = 3
```

#### `max_misses_M`
**Type:** Integer
**Default:** Varies by tracker
**Range:** `1-120`

Maximum number of consecutive missed detections before a track is deleted.

**TOML:**
```toml
[tracking.params]
max_misses_M = 15
```

#### `offline_linker_max_misses`
**Type:** Integer
**Default:** `30`
**Range:** `1-600`

Maximum frame gap the offline linker will attempt to bridge.

**TOML:**
```toml
[tracking.params]
offline_linker_max_misses = 45
```

#### `offline_linker_per_frame_gate`
**Type:** Float
**Default:** `0.05`
**Range:** `0.0-1.0`

Maximum per-frame distance threshold for offline linking.

**TOML:**
```toml
[tracking.params]
offline_linker_per_frame_gate = 0.03
```

#### `bbox_dilate_pct`
**Type:** Float
**Default:** Varies by tracker
**Range:** `0.0-0.6`

Percentage to expand bounding boxes (for tracking stability).

**TOML:**
```toml
[tracking.params]
bbox_dilate_pct = 0.25
```

#### `temporal_smooth_alpha`
**Type:** Float
**Default:** Varies by tracker
**Range:** `0.0-1.0`

Temporal smoothing factor (1.0 = no smoothing, 0.0 = maximum smoothing).

**TOML:**
```toml
[tracking.params]
temporal_smooth_alpha = 0.7
```

### ByteTrack-Specific Parameters

#### `use_low_score_pool`
**Type:** Boolean
**Default:** `true`

Enable low-score detection pool for second-chance matching.

#### `high_thresh`
**Type:** Float
**Default:** `0.4`
**Range:** `0.0-1.0`

High confidence threshold for first-stage matching.

#### `low_thresh`
**Type:** Float
**Default:** `0.1`
**Range:** `0.0-1.0`

Low confidence threshold for second-stage matching.

**Example TOML:**
```toml
[tracking]
type = "bytetrack"

[tracking.params]
distance_gate = 0.05
high_thresh = 0.5
low_thresh = 0.2
use_low_score_pool = true
```

### BotSort-Specific Parameters

#### `cam_motion_comp`
**Type:** Boolean
**Default:** `true`

Enable camera motion compensation.

#### `flow_backend`
**Type:** String
**Default:** `"LK"`

Optical flow backend for motion estimation.

#### `distance_gate_hi`
**Type:** Float
**Default:** `0.05`

Distance gate for high-confidence detections.

#### `distance_gate_lo`
**Type:** Float
**Default:** `0.02`

Distance gate for low-confidence detections.

**Example TOML:**
```toml
[tracking]
type = "botsort"

[tracking.params]
cam_motion_comp = true
flow_backend = "LK"
distance_gate_hi = 0.05
distance_gate_lo = 0.02
high_thresh = 0.4
low_thresh = 0.1
```

### Hybrid SOT-Specific Parameters

#### `use_visual_tracker`
**Type:** Boolean
**Default:** `true`

Enable visual single-object tracking for missed detections.

#### `vt_backend`
**Type:** String
**Default:** `"CSRT"`

Visual tracker backend (OpenCV tracker algorithm).

#### `vt_max_age`
**Type:** Integer
**Default:** `10`
**Range:** `0-120`

Maximum age (frames) before visual tracker is dropped.

#### `drift_gate`
**Type:** Float
**Default:** `0.05`
**Range:** `0.0-2.0`

Maximum drift threshold for visual tracker.

**Example TOML:**
```toml
[tracking]
type = "hybrid_sot"

[tracking.params]
use_visual_tracker = true
vt_backend = "CSRT"
vt_max_age = 10
drift_gate = 0.05
distance_gate = 0.05
```

---

## Video Configuration

### `video.codec`
**Type:** String
**Default:** `"h264"`
**Choices:** `h264`, `hevc`, `vp8`, `vp9`

Output video codec.

**Recommendations:**
- **h264** - Best compatibility, widely supported
- **hevc** - Better compression, smaller files, less compatible
- **vp8/vp9** - Open source, good for web

**TOML:**
```toml
[video]
codec = "hevc"
```

**CLI:**
```bash
blur-cli video input.mp4 --video-codec hevc
```

### `video.quality`
**Type:** Integer or null
**Default:** `null` (uses encoder default)
**Range:** `1-51`

Video quality setting (CRF for H.264/HEVC). Lower values = better quality, larger file size.

**Recommendations:**
- High quality: `18-23`
- Medium quality: `23-28`
- Low quality: `28-35`

**TOML:**
```toml
[video]
quality = 23
```

**CLI:**
```bash
blur-cli video input.mp4 --video-quality 23
```

---

## Environment variables

Obscuro no longer supports overriding arbitrary config fields via `BLUR_*` environment variables. Use TOML/CLI options instead.
Only a handful of path/launch settings remain environment-driven:

| Variable | Purpose |
| --- | --- |
| `BLUR_DATA_DIR` | Override the root data directory used by the backend and CLI (defaults to the OS-specific user-data path). |
| `BLUR_MODELS_DIR` | Override the directory where ONNX models are stored/discovered. |
| `BLUR_BACKEND_AUTOSTART` | When running the Electron app, force the backend launcher to `uv`, `docker`, or `auto`. |
| `BLUR_BACKEND_DOCKER_IMAGE_BASE` / `BLUR_BACKEND_DOCKER_CPU_IMAGE` / `BLUR_BACKEND_DOCKER_GPU_IMAGE` | Customize which Docker images the Electron app uses when auto-starting the backend. |
| `BLUR_BACKEND_ROOT` | Explicitly point the Electron backend manager to a checkout when the sources are not bundled. |

Example (custom models directory):

```bash
export BLUR_MODELS_DIR=/custom/path/to/models
uv run blur-cli models
```

---

## Complete Configuration Examples

### High-Quality Processing

```toml
debug = false
log_level = "INFO"

[model]
name = "1280_nano"

[blur]
type = "gaussian"
strength = 15

[detection]
plate_threshold = 0.4
face_threshold = 0.4
batch_size = 8
use_sahi = true
inference_size = 1920

[tracking]
type = "botsort"
use_offline_linker = true

[tracking.params]
confirm_after_N = 3
max_misses_M = 8
cam_motion_comp = true
high_thresh = 0.5
low_thresh = 0.2

[video]
codec = "h264"
quality = 20
```

### Fast Processing (Speed Priority)

```toml
[model]
name = "640_nano"

[blur]
type = "blackout"
strength = 5

[detection]
plate_threshold = 0.5
face_threshold = 0.5
batch_size = 16
use_sahi = false
inference_size = 640

[tracking]
type = "dummy"  # No tracking
use_offline_linker = false

[video]
codec = "h264"
quality = 28
```

### 4K High-Resolution Processing

```toml
[model]
name = "1280_nano"

[blur]
type = "gaussian"
strength = 20

[detection]
plate_threshold = 0.3
face_threshold = 0.3
batch_size = 4
use_sahi = true
inference_size = 3840
sahi_overlap_ratio = 0.25

[tracking]
type = "botsort"
use_offline_linker = true

[tracking.params]
max_misses_M = 15
offline_linker_max_misses = 60

[video]
codec = "hevc"
quality = 22
```

### Privacy-Focused (Maximum Blur)

```toml
[blur]
type = "blackout"
strength = 100

[detection]
plate_threshold = 0.2  # Low threshold, catch everything
face_threshold = 0.2
batch_size = 4
use_sahi = true
inference_size = 2560

[tracking]
type = "bytetrack"
use_offline_linker = true

[tracking.params]
confirm_after_N = 1  # Blur immediately
max_misses_M = 30     # Keep blurring even if detection lost
```

---

## Model Storage Location

Models are stored in platform-specific directories:

- **macOS**: `~/Library/Application Support/blur_gui/models`
- **Linux**: `~/.local/share/blur_gui/models` (or `$XDG_DATA_HOME/blur_gui/models`)
- **Windows**: `%LOCALAPPDATA%\blur_gui\models`

Override with environment variable:

```bash
export BLUR_MODELS_DIR=/custom/path/to/models
```

---

## Tips for Optimal Configuration

### GPU Memory Optimization

If you encounter out-of-memory errors:

1. Reduce `detection.batch_size` to 2-4
2. Lower `detection.inference_size` to 1280 or 1536
3. Disable SAHI if enabled

### Tracking Quality

For better tracking quality:

1. Use `tracker = "botsort"` for best results
2. Keep `use_offline_linker = true`
3. Increase `max_misses_M` to 15-30 for crowded scenes
4. Lower detection thresholds to 0.3-0.4

### Processing Speed

For faster processing:

1. Use `tracker = "dummy"` to skip tracking
2. Set `use_offline_linker = false`
3. Increase `batch_size` if GPU memory allows
4. Use `blur_type = "blackout"` (fastest)
5. Disable SAHI

### Privacy vs Quality

Balance privacy protection with visual quality:

- **Maximum privacy**: Low thresholds (0.2-0.3), high blur strength (20+), `blackout` type
- **Balanced**: Default thresholds (0.5), medium blur (10-15), `gaussian` type
- **Minimal impact**: High thresholds (0.6-0.7), low blur (5-10), `gaussian` type

### High-Resolution Videos

For 4K or higher resolution videos:

1. Enable SAHI: `use_sahi = true`
2. Set `inference_size = 2560` or higher
3. Increase `sahi_overlap_ratio = 0.25-0.3`
4. Reduce `batch_size` if memory limited
5. Use HEVC codec for smaller output files

---

## Configuration Validation

The system validates all configuration options and will report errors for:

- Invalid ranges (e.g., `blur_strength = 150`)
- Unknown options
- Type mismatches
- Missing required model files

Validation errors are reported at startup with clear error messages.
