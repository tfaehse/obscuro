"""
Shared fixtures and configuration for all tests.
"""

import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from anonymizer.config import (
    DEFAULT_TRACKER_PARAMS,
    AnonymizerConfig,
    BlurConfig,
    BlurType,
    DetectionConfig,
    ModelConfig,
    TrackerType,
    TrackingConfig,
    VideoConfig,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model_path(temp_dir):
    """Create a mock model file for testing."""
    model_path = temp_dir / "test_model.onnx"
    model_path.write_bytes(b"fake_model_data")
    return model_path


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return AnonymizerConfig(
        model=ModelConfig(name="test_model"),
        blur=BlurConfig(type=BlurType.GAUSSIAN, strength=10),
        detection=DetectionConfig(confidence_threshold=0.5, low_score_threshold=0.1),
        tracking=TrackingConfig(
            type=TrackerType.BYTETRACK,
            params=dict(DEFAULT_TRACKER_PARAMS[TrackerType.BYTETRACK]),
        ),
        video=VideoConfig(codec="h264"),
    )


@pytest.fixture
def mock_progress_callback():
    """Create a mock progress callback."""
    return Mock()


@pytest.fixture
def cancel_event():
    """Create a threading event for cancellation."""
    return threading.Event()


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_video_frames():
    """Create sample video frames for testing."""
    frames = []
    for _ in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


@pytest.fixture
def sample_detections():
    """Create sample detection data."""
    return {
        "faces": [
            {"x": 100, "y": 100, "width": 50, "height": 50, "confidence": 0.8},
            {"x": 200, "y": 150, "width": 45, "height": 45, "confidence": 0.9},
        ],
        "plates": [
            {"x": 300, "y": 300, "width": 80, "height": 30, "confidence": 0.7},
        ],
    }


@pytest.fixture
def sample_blur_rois():
    """Create sample blur ROI data."""
    return [
        {"x": 100, "y": 100, "width": 50, "height": 50, "track_id": 1, "frame": 0},
        {"x": 105, "y": 105, "width": 50, "height": 50, "track_id": 1, "frame": 1},
        {"x": 300, "y": 300, "width": 80, "height": 30, "track_id": 2, "frame": 0},
    ]


@pytest.fixture
def mock_onnx_model():
    """Create a mock ONNX model."""
    mock_model = Mock()
    mock_model.get_inputs.return_value = [Mock(name="input", shape=[1, 3, 640, 640])]
    mock_model.get_outputs.return_value = [Mock(name="output")]
    return mock_model


@pytest.fixture
def mock_video_file(temp_dir):
    """Create a mock video file."""
    video_path = temp_dir / "test_video.mp4"
    # Create a minimal valid video file structure (just touch the file for basic tests)
    video_path.touch()
    return video_path


@pytest.fixture
def mock_config_file(temp_dir):
    """Create a mock configuration file."""
    config_path = temp_dir / "test_config.toml"
    config_content = """
[model]
name = "test_model"

[blur]
type = "gaussian"
strength = 10

[detection]
confidence_threshold = 0.5
low_score_threshold = 0.1

[tracking]
type = "bytetrack"
use_offline_linker = true

[tracking.params]
bbox_dilate_pct = 0.2
temporal_smooth_alpha = 0.6
ema_alpha = 0.6
distance_gate = 0.4
confirm_after_N = 2
max_misses_M = 10
offline_linker_max_misses = 30
offline_linker_per_frame_gate = 0.05
confidence_threshold = 0.6
low_score_threshold = 0.2
use_low_score_pool = true
distance_gate_hi = 0.05
distance_gate_lo = 0.01
cam_motion_comp = true
flow_backend = "LK"
use_visual_tracker = false
vt_backend = "TrackerNano"
vt_max_age = 6
drift_gate = 0.15
process_noise = 1.0

[video]
codec = "h264"
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset global configuration after each test."""
    yield
    from anonymizer import config

    # Reset global config instance to None
    config._config_instance = None
