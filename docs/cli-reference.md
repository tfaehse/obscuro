# CLI Reference

The `blur-cli` command-line interface provides access to all Obscuro functionality for batch processing and automation.

## Installation

After installing Obscuro, the CLI is available as `blur-cli`:

```bash
uv sync  # Install dependencies
blur-cli --help  # Show help
```

## Global Options

These options apply to all commands:

```bash
blur-cli [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

### `--config PATH`
Load configuration from a TOML file.

```bash
blur-cli --config my_config.toml video input.mp4
```

### `--log-level LEVEL`
Set logging verbosity. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

Default: `INFO`

```bash
blur-cli --log-level DEBUG video input.mp4
```

### `--json-log PATH`
Enable JSON-formatted logging to the specified file.

```bash
blur-cli --json-log processing.json video input.mp4
```

### `--no-colors`
Disable colored terminal output.

```bash
blur-cli --no-colors video input.mp4
```

### `--config-debug` / `--no-config-debug`
Enable or disable debug mode in the anonymizer configuration.

```bash
blur-cli --config-debug video input.mp4
```

## Commands

### `image` - Process Images

Blur faces and license plates in a single image.

```bash
blur-cli image INPUT [OPTIONS]
```

#### Arguments

- `INPUT` - Path to input image file (required)

#### Options

##### `-o, --output PATH`
Output image file path. If not specified, auto-generates a filename with `_blurred` suffix.

```bash
blur-cli image input.jpg -o output.jpg
```

##### Model Options

###### `--model NAME_OR_PATH`
Specify which model to use. Can be:
- Model name (without .onnx extension): `--model 1280_nano` or `--model 640_nano`
- Full path to ONNX file: `--model /path/to/model.onnx`

Overrides config file setting.

```bash
blur-cli image input.jpg --model 1280_nano
```

##### Blur Options

###### `--blur-type TYPE`
Type of blur to apply. Choices:
- `gaussian` - Smooth Gaussian blur (default)
- `pixelate` - Pixelated/mosaic effect
- `blackout` - Solid black box
- `debug` - Shows detection boxes without blurring

```bash
blur-cli image input.jpg --blur-type pixelate
```

###### `--blur-strength N`
Blur strength/intensity (1-100). Higher values = stronger blur.

Default: `10`

```bash
blur-cli image input.jpg --blur-strength 25
```

##### Detection Options

###### `--plate-threshold FLOAT`
Confidence threshold for license plate detection (0.0-1.0). Lower values detect more plates but may include false positives.

Default: `0.5`

```bash
blur-cli image input.jpg --plate-threshold 0.3
```

###### `--face-threshold FLOAT`
Confidence threshold for face detection (0.0-1.0).

Default: `0.5`

```bash
blur-cli image input.jpg --face-threshold 0.6
```

###### `--blur-classes LIST`
Comma-separated list of detector classes to blur. Supported values: `person`, `car`, `bus`, `motorcycle`, `truck`.

```bash
blur-cli video input.mp4 --blur-classes person,car,truck
```

###### `--batch-size N`
Number of frames/images processed per inference call. Higher values can improve GPU throughput but use more memory.

Default: `4`

```bash
blur-cli image input.jpg --batch-size 8
```

> Models whose filenames end with `_b1` are fixed-batch exports and always run with batch size 1. The CLI will log a notice and override larger values automatically. The reason is that CoreML on macOS currently does not support dynamic batch sizes, and the speedup over pure CPU inference can be significant even at batch size 1. Usage is only recommended on Apple Silicon hardware.

###### `--use-sahi` / `--no-use-sahi`
Enable or disable SAHI (Slicing Aided Hyper Inference) for tiled inference. Useful for high-resolution images.

Default: disabled

```bash
blur-cli image large_input.jpg --use-sahi
```

###### `--inference-size N`
Longest image edge in pixels used for detection inference (256-8192). Images larger than this are downscaled before detection.

Default: `1920`

```bash
blur-cli image input.jpg --inference-size 2560
```

###### `--sahi-overlap FLOAT`
SAHI tile overlap ratio (0.0-0.99). Only used when `--use-sahi` is enabled.

Default: `0.2`

```bash
blur-cli image input.jpg --use-sahi --sahi-overlap 0.3
```

##### Tracking Options

###### `--tracker TYPE`
Tracker algorithm to use. Choices:
- `dummy` - No tracking, frame-by-frame detection only
- `bytetrack` - Fast ByteTrack algorithm (default)
- `botsort` - BoT-SORT with motion compensation
- `fused` - ByteTrack-style association with distance + shape + embeddings
- `hybrid_sot` - Fused tracker plus per-track visual tracker for missed detections

```bash
blur-cli image input.jpg --tracker botsort
```

###### `--tracker-params JSON`
JSON object with tracker parameter overrides. See [Configuration Guide](configuration.md#tracker-parameters) for available parameters.

```bash
blur-cli image input.jpg --tracker-params '{"distance_gate":0.15,"max_misses_M":5}'
```

###### `--embedding-similarity-gate FLOAT`
Override the minimum embedding cosine similarity used by `fused`/`hybrid_sot` trackers.

```bash
blur-cli video input.mp4 --embedding-similarity-gate 0.6
```

###### `--offline-linker` / `--no-offline-linker`
Enable or disable the offline tracklet linker (post-processing pass to reconnect broken tracks).

Default: enabled

```bash
blur-cli image input.jpg --no-offline-linker
```

##### Video Options

###### `--video-codec CODEC`
Output video codec. Choices: `h264`, `hevc`, `vp8`, `vp9`

Default: `h264`

```bash
blur-cli video input.mp4 --video-codec hevc
```

###### `--video-quality N`
Video quality setting (1-51). Lower values = better quality, larger file size.

Default: `None` (uses encoder default)

```bash
blur-cli video input.mp4 --video-quality 23
```

#### Examples

```bash
# Basic image blur
blur-cli image photo.jpg

# High-resolution image with SAHI
blur-cli image large_photo.jpg --use-sahi --inference-size 3840

# Custom blur with lower thresholds
blur-cli image input.jpg --blur-type pixelate --blur-strength 30 \
  --plate-threshold 0.3 --face-threshold 0.4

# Specify output location
blur-cli image input.jpg -o results/blurred.jpg
```

---

### `video` - Process Videos

Blur faces and license plates in video files.

```bash
blur-cli video INPUT [OPTIONS]
```

#### Arguments

- `INPUT` - Path to input video file (required)

#### Options

Same as [`image` command](#image---process-images), plus:

##### `-o, --output PATH`
Output video file path. If not specified, auto-generates a filename with `_blurred` suffix.

```bash
blur-cli video input.mp4 -o output.mp4
```

#### Examples

```bash
# Basic video blur
blur-cli video dashcam.mp4

# High-quality processing with BotSort tracking
blur-cli video dashcam.mp4 --tracker botsort --video-quality 18

# Fast processing with pixelation
blur-cli video dashcam.mp4 --blur-type pixelate --tracker dummy

# Custom tracking parameters
blur-cli video dashcam.mp4 --tracker bytetrack \
  --tracker-params '{"distance_gate":0.05,"confirm_after_N":3}'

# Use custom model
blur-cli video dashcam.mp4 --model /path/to/custom_model.onnx
```

---

### `config` - Generate Configuration Template

Create a TOML configuration file template with all available options.

```bash
blur-cli config [OPTIONS]
```

#### Options

##### `-o, --output PATH`
Output configuration file path.

Default: `blur_config.toml`

```bash
blur-cli config -o my_config.toml
```

#### Example

```bash
# Generate config template
blur-cli config -o production.toml

# Edit the file
nano production.toml

# Use the config
blur-cli --config production.toml video input.mp4
```

---

### `models` - Manage Models

List available ONNX detection models or download new ones. Files are read from `<models root>/detection`.

```bash
blur-cli models [OPTIONS]
```

#### Options

##### `--download URL`
Download an ONNX model from the specified URL.

```bash
blur-cli models --download https://example.com/model.onnx
```

##### `--name NAME`
Optional name for the downloaded model (without .onnx extension). If not specified, uses the filename from the URL.

```bash
blur-cli models --download https://example.com/my_model.onnx --name custom_model
```

#### Examples

```bash
# List available models
blur-cli models

# Download a model
blur-cli models --download https://example.com/1280_nano.onnx --name 1280_nano

# List models after download
blur-cli models
```

## Configuration Priority

Configuration is loaded in the following priority order (later sources override earlier):

1. Built-in defaults
2. TOML configuration file (specified with `--config`)
3. Environment variables
4. Command-line arguments

## Model Storage

Models are stored in platform-specific directories (with `detection/` and `tracking/` subfolders inside):

- **macOS**: `~/Library/Application Support/blur_gui/models`
- **Linux**: `~/.local/share/blur_gui/models`
- **Windows**: `%LOCALAPPDATA%\blur_gui\models`

Visual tracker weights for `TrackerNano` are **not bundled**. Download the official backbone/neckhead ONNX files yourself and place them under `<models root>/tracking` if you enable the `hybrid_sot` visual tracker backend.

Override with the `BLUR_MODELS_DIR` environment variable:

```bash
BLUR_MODELS_DIR=/custom/path blur-cli models
```

## Progress Tracking

The CLI displays real-time progress during video processing:

```
Progress: 25.5% - Detection: Processing frame 128/500
Progress: 45.2% - Tracking: Associating detections
Progress: 78.3% - Blurring: Applying effects
```

## Error Handling

The CLI returns exit codes:

- `0` - Success
- `1` - Error occurred

Check the exit code in scripts:

```bash
blur-cli video input.mp4
if [ $? -eq 0 ]; then
    echo "Success!"
else
    echo "Failed!"
fi
```

## Tips

- Use `--json-log` for machine-readable logs that can be parsed by monitoring tools
- Enable `--config-debug` to see detailed processing information
- For large videos, increase `--batch-size` if you have GPU memory available
- Use `--tracker dummy` for fastest processing when tracking quality isn't critical
- Enable `--use-sahi` for videos with high resolution or distant objects
