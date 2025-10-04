# API Reference

The Obscuro FastAPI server provides a REST API for programmatic video anonymization.

## Starting the API Server

### Using the convenience wrapper

```bash
blur-api --host 0.0.0.0 --port 8000
```

### Using uvicorn directly

```bash
uvicorn blur_api.serve:app --reload --host 0.0.0.0 --port 8000
```

### Server Options

#### `--host HOST`
Host address to bind to.

Default: `0.0.0.0`

#### `--port PORT`
Port to bind to.

Default: `8000`

#### `--log-level LEVEL`
Logging level. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

Default: `INFO`

#### `--json-log PATH`
Enable JSON logging to the specified file.

#### `--reload`
Enable auto-reload for development (watches for file changes).

#### `--config PATH`
Path to TOML configuration file to load before starting.

### Docker

```bash
# Build the CPU image locally (optional if you pull from GHCR)
docker build -t ghcr.io/<your-account>/blur-gui-backend:latest .

# Run the container
docker run --rm -p 8000:8000 \
  -e BLUR_MODELS_DIR=/data/models \
  -v "$(pwd)/models:/data/models" \
  ghcr.io/<your-account>/blur-gui-backend:latest
```

Published builds are available on GitHub Container Registry via the release workflow:

```bash
# Authenticate once (public images can skip --password-stdin if anonymous pull is enabled)
echo "${GHCR_TOKEN}" | docker login ghcr.io -u <your-account> --password-stdin

# Pull the pre-built backend
docker pull ghcr.io/<your-account>/blur-gui-backend:latest
docker pull ghcr.io/<your-account>/blur-gui-backend:gpu
```

Use the GPU tag together with `--gpus all` if you have the NVIDIA container runtime installed.

## Base URL

All endpoints are relative to the server URL:

```
http://localhost:8000
```

## Endpoints

### Configuration Endpoints

#### `GET /blur/config/options`

Get available configuration options and current settings.

**Response:**

```json
{
  "model": {
    "available": ["1280_nano", "640_nano"],
    "current": "1280_nano",
    "files": [
      {
        "name": "1280_nano",
        "filename": "1280_nano.onnx",
        "size_bytes": 12345678,
        "modified_at": "2025-11-06T10:30:00Z",
        "immutable": true
      }
    ]
  },
  "blur": {
    "types": ["gaussian", "pixelate", "blackout", "black", "debug"],
    "current_type": "gaussian",
    "current_strength": 10,
    "strength_range": [1, 100]
  },
  "detection": {
    "current_plate_threshold": 0.5,
    "current_face_threshold": 0.5,
    "current_batch_size": 4,
    "threshold_range": [0.0, 1.0],
    "use_sahi": true,
    "current_inference_size": 1920,
    "inference_size_range": [256, 8192],
    "current_sahi_overlap": 0.2,
    "sahi_overlap_range": [0.0, 0.99]
  },
  "tracking": {
    "types": ["dummy", "bytetrack", "botsort", "hybrid_sot"],
    "current_type": "bytetrack",
    "params": {
      "distance_gate": 0.05,
      "confirm_after_N": 2,
      "max_misses_M": 10,
      ...
    },
    "use_offline_linker": true,
    "ranges": {
      "distance_gate": [0.05, 1.0],
      "confirm_after_N": [1, 5],
      "max_misses_M": [1, 30],
      ...
    }
  },
  "video": {
    "codecs": ["h264", "hevc", "vp8", "vp9"],
    "current_codec": "h264",
    "current_quality": null,
    "quality_range": [1, 51]
  },
  "global": {
    "log_levels": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    "current_debug": false,
    "current_log_level": "INFO"
  }
}
```

#### `GET /blur/config/current`

Get the current active configuration.

**Response:**

```json
{
  "model": {
    "name": "1280_nano",
    "file": null
  },
  "blur": {
    "type": "gaussian",
    "strength": 10
  },
  "detection": {
    "plate_threshold": 0.5,
    "face_threshold": 0.5,
    "batch_size": 4,
    "use_sahi": true,
    "inference_size": 1920,
    "sahi_overlap_ratio": 0.2
  },
  "tracking": {
    "type": "bytetrack",
    "params": {...},
    "use_offline_linker": true
  },
  "video": {
    "codec": "h264",
    "quality": null
  },
  "debug": false,
  "log_level": "INFO"
}
```

---

### Model Management Endpoints

#### `GET /blur/models`

List all available ONNX models.

**Response:**

```json
{
  "models": [
    {
      "name": "1280_nano",
      "filename": "1280_nano.onnx",
      "size_bytes": 12345678,
      "modified_at": "2025-11-06T10:30:00Z",
      "immutable": true
    }
  ]
}
```

#### `POST /blur/models`

Upload a new ONNX model.

**Request:**

- Content-Type: `multipart/form-data`
- Fields:
  - `file` (file, required) - ONNX model file
  - `name` (string, optional) - Model name (without .onnx extension)

**Example using curl:**

```bash
curl -X POST http://localhost:8000/blur/models \
  -F "file=@my_model.onnx" \
  -F "name=custom_model"
```

**Response:**

```json
{
  "models": [...],
  "added": {
    "name": "custom_model",
    "filename": "custom_model.onnx",
    "size_bytes": 12345678,
    "modified_at": "2025-11-06T10:35:00Z"
  }
}
```

#### `DELETE /blur/models/{model_name}`

Delete a model by name.

**Parameters:**

- `model_name` (path) - Model name (with or without .onnx extension)

**Example:**

```bash
curl -X DELETE http://localhost:8000/blur/models/custom_model
```

**Response:**

```json
{
  "models": [...]
}
```

---

### Processing Endpoints

#### `POST /blur/frame`

Blur a single frame/image (for real-time preview).

**Request:**

- Content-Type: `multipart/form-data`
- Fields:
  - `input` (file, required) - Image file
  - `config` (string, optional) - JSON configuration overrides

**Configuration Override Format:**

The `config` field accepts a JSON string with nested configuration overrides:

```json
{
  "blur": {
    "type": "pixelate",
    "strength": 20
  },
  "detection": {
    "plate_threshold": 0.3,
    "face_threshold": 0.4
  }
}
```

**Example using curl:**

```bash
curl -X POST http://localhost:8000/blur/frame \
  -F "input=@frame.jpg" \
  -F 'config={"blur":{"type":"pixelate","strength":25}}'
```

**Response:**

- Content-Type: `image/jpeg`
- Body: Blurred image as JPEG

**Note:** This endpoint automatically uses CPU processing if GPU is busy, making it suitable for real-time preview while video processing is running.

#### `POST /blur/image_file`

Blur an image file on the server's filesystem.

**Request:**

- Content-Type: `multipart/form-data`
- Fields:
  - `input` (string, required) - Path to input image file on server
  - `output_path` (string, required) - Path to save output image on server
  - `config` (string, optional) - JSON configuration overrides

**Example using curl:**

```bash
curl -X POST http://localhost:8000/blur/image_file \
  -F "input=/path/to/input.jpg" \
  -F "output_path=/path/to/output.jpg" \
  -F 'config={}'
```

**Response:**

```json
{
  "message": "Image blurred and saved successfully."
}
```

#### `POST /blur/video_file`

Start asynchronous video processing job.

**Request:**

- Content-Type: `multipart/form-data`
- Fields:
  - `input` (file, required) - Video file
  - `config` (string, optional) - JSON configuration overrides
  - `output_filename` (string, optional) - Desired output filename

**Example using curl:**

```bash
curl -X POST http://localhost:8000/blur/video_file \
  -F "input=@dashcam.mp4" \
  -F 'config={"blur":{"type":"pixelate"},"tracking":{"type":"botsort"}}' \
  -F "output_filename=result.mp4"
```

**Response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Use the `job_id` to track progress and download results.

#### `GET /blur/video_progress/{job_id}`

Get real-time progress updates for a video processing job (Server-Sent Events).

**Parameters:**

- `job_id` (path) - Job ID returned from `/blur/video_file`

**Response:**

Stream of Server-Sent Events with JSON data:

```
data: {"job_id":"550e8400-...","progress":25,"message":"Detection: Processing frame 128/500","stage":"Detection","stage_message":"Processing frame 128/500","status":"running","error":null,"sequence":15,"updated_at":1699272345.123,"output_path":null}

data: {"job_id":"550e8400-...","progress":50,"message":"Tracking: Associating detections","stage":"Tracking","stage_message":"Associating detections","status":"running","error":null,"sequence":16,"updated_at":1699272350.456,"output_path":null}

data: {"job_id":"550e8400-...","progress":100,"message":"Blurring: Complete","stage":"Blurring","stage_message":"Complete","status":"done","error":null,"sequence":20,"updated_at":1699272360.789,"output_path":"/tmp/obscuro_jobs/session_abc123/550e8400-..._output.mp4"}
```

**Job Status Values:**

- `running` - Job is processing
- `done` - Job completed successfully
- `error` - Job failed
- `cancelled` - Job was cancelled

**Example using curl:**

```bash
curl -N http://localhost:8000/blur/video_progress/550e8400-e29b-41d4-a716-446655440000
```

**Example using JavaScript:**

```javascript
const eventSource = new EventSource('/blur/video_progress/550e8400-...');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress}%`);
  console.log(`Status: ${data.status}`);

  if (data.status === 'done') {
    eventSource.close();
    // Download the result
  }
};
```

#### `POST /blur/cancel_video/{job_id}`

Cancel a running video processing job.

**Parameters:**

- `job_id` (path) - Job ID to cancel

**Example:**

```bash
curl -X POST http://localhost:8000/blur/cancel_video/550e8400-e29b-41d4-a716-446655440000
```

**Response:**

```json
{
  "message": "Job cancelled successfully"
}
```

**Error Responses:**

- `404 Not Found` - Job ID not found
- `400 Bad Request` - Job cannot be cancelled (already completed)

#### `GET /blur/download/{job_id}`

Download the processed video result.

**Parameters:**

- `job_id` (path) - Job ID

**Response:**

- Content-Type: `video/mp4`
- Content-Disposition: `attachment; filename="blurred_video_{job_id}.mp4"`
- Body: Processed video file

**Example:**

```bash
curl -O -J http://localhost:8000/blur/download/550e8400-e29b-41d4-a716-446655440000
```

**Note:** The job is automatically cleaned up after download, and the temporary files are deleted.

**Error Responses:**

- `404 Not Found` - Job ID not found or output file missing
- `400 Bad Request` - Job not completed yet

---

### Health Check

#### `GET /healthz`

#### `HEAD /healthz`

Check API health and backend status.

**Response:**

```json
{
  "status": "ok",
  "status_code": 0,
  "execution_provider": "CUDAExecutionProvider",
  "requested_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "active_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
}
```

**Status Values:**

- `ok` (status_code: 0) - Backend is ready and GPU available
- `degraded` (status_code: 1) - Backend running but GPU not available (using CPU)
- `busy` (status_code: 1) - Backend is currently processing
- `error` (status_code: -1) - Backend initialization error

**Example:**

```bash
curl http://localhost:8000/healthz
```

## Configuration Overrides

Most processing endpoints accept a `config` parameter for runtime overrides. The configuration must be a **nested JSON object** matching the `AnonymizerConfig` structure:

### Example Configuration Override

```json
{
  "model": {
    "name": "1280_nano"
  },
  "blur": {
    "type": "pixelate",
    "strength": 25
  },
  "detection": {
    "plate_threshold": 0.3,
    "face_threshold": 0.4,
    "batch_size": 8,
    "use_sahi": true,
    "inference_size": 2560,
    "sahi_overlap_ratio": 0.25
  },
  "tracking": {
    "type": "botsort",
    "use_offline_linker": true,
    "params": {
      "distance_gate": 0.1,
      "max_misses_M": 8
    }
  },
  "video": {
    "codec": "h264",
    "quality": 23
  },
  "debug": false,
  "log_level": "INFO"
}
```

You can override any subset of these values. Unspecified fields use the server's current configuration.

## Error Responses

All endpoints return standard HTTP error codes:

- `400 Bad Request` - Invalid input or configuration
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Processing error

Error response format:

```json
{
  "detail": "Error message describing the issue"
}
```

## Rate Limiting

The API does not currently implement rate limiting, but be aware that:

- Only one GPU-backed video job can run at a time
- Frame preview requests use CPU when GPU is busy
- Multiple concurrent requests may impact performance

## CORS

The API allows cross-origin requests from any origin (`*`). Configure `allow_origins` in production for security.

## Complete Workflow Example

### Python Example

```python
import requests
import json
import time

API_BASE = "http://localhost:8000"

# 1. Check available models
response = requests.get(f"{API_BASE}/blur/config/options")
options = response.json()
print(f"Available models: {options['model']['available']}")

# 2. Upload a video and start processing
config = {
    "blur": {"type": "pixelate", "strength": 20},
    "tracking": {"type": "botsort"}
}

with open("dashcam.mp4", "rb") as f:
    files = {"input": f}
    data = {
        "config": json.dumps(config),
        "output_filename": "result.mp4"
    }
    response = requests.post(f"{API_BASE}/blur/video_file", files=files, data=data)

job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")

# 3. Poll for progress
while True:
    response = requests.get(f"{API_BASE}/blur/video_progress/{job_id}")
    # Note: This is simplified; use SSE for real-time updates
    time.sleep(2)
    # Check if done (see SSE example above for proper implementation)

# 4. Download result
response = requests.get(f"{API_BASE}/blur/download/{job_id}")
with open("blurred_output.mp4", "wb") as f:
    f.write(response.content)

print("Processing complete!")
```

### JavaScript/TypeScript Example

```typescript
const API_BASE = 'http://localhost:8000';

async function processVideo(file: File) {
  // 1. Upload and start processing
  const formData = new FormData();
  formData.append('input', file);
  formData.append('config', JSON.stringify({
    blur: { type: 'pixelate', strength: 20 },
    tracking: { type: 'botsort' }
  }));

  const response = await fetch(`${API_BASE}/blur/video_file`, {
    method: 'POST',
    body: formData
  });

  const { job_id } = await response.json();
  console.log(`Job ID: ${job_id}`);

  // 2. Monitor progress with SSE
  const eventSource = new EventSource(`${API_BASE}/blur/video_progress/${job_id}`);

  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`Progress: ${data.progress}% - ${data.message}`);

    if (data.status === 'done') {
      eventSource.close();
      downloadResult(job_id);
    } else if (data.status === 'error') {
      eventSource.close();
      console.error(`Error: ${data.error}`);
    }
  };
}

async function downloadResult(job_id: string) {
  const response = await fetch(`${API_BASE}/blur/download/${job_id}`);
  const blob = await response.blob();

  // Trigger download
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `blurred_${job_id}.mp4`;
  a.click();
  window.URL.revokeObjectURL(url);
}
```

## Tips

- Use `/blur/frame` for real-time preview before processing full videos
- Monitor `/healthz` to check GPU availability before submitting jobs
- Store uploaded models persistently using Docker volumes
- Use Server-Sent Events for real-time progress updates
- Cancel long-running jobs with `/blur/cancel_video/{job_id}`
- Adjust `batch_size` and `inference_size` for optimal GPU utilization
