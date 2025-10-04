"""
Tests for blur_api.serve.

This module contains tests for the FastAPI server endpoints and helper functions
that handle video/image anonymization requests.
"""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from anonymizer.config import AnonymizerConfig, BlurType
from blur_api import serve
from blur_api.serve import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_anonymizer_instances():
    """Ensure GPU/CPU anonymizer globals do not leak across tests."""
    snapshot = {
        "gpu_anonymizer_instance": serve.gpu_anonymizer_instance,
        "cpu_anonymizer_instance": serve.cpu_anonymizer_instance,
        "active_gpu_model_key": serve.active_gpu_model_key,
        "active_cpu_model_key": serve.active_cpu_model_key,
    }
    serve.gpu_anonymizer_instance = None
    serve.cpu_anonymizer_instance = None
    serve.active_gpu_model_key = None
    serve.active_cpu_model_key = None
    yield
    serve.gpu_anonymizer_instance = snapshot["gpu_anonymizer_instance"]
    serve.cpu_anonymizer_instance = snapshot["cpu_anonymizer_instance"]
    serve.active_gpu_model_key = snapshot["active_gpu_model_key"]
    serve.active_cpu_model_key = snapshot["active_cpu_model_key"]


def test_get_model_key_defaults():
    """Test model key generation with default configuration."""
    config = AnonymizerConfig()
    key = serve.get_model_key(config)
    assert isinstance(key, str)
    assert "-" in key  # Key should contain delimiter


def test_get_model_key_custom():
    """Test model key reflects custom configuration values."""
    config = AnonymizerConfig()
    config.model.name = "foo"
    config.blur.type = BlurType("pixelate")
    config.blur.strength = 5
    key = serve.get_model_key(config)
    assert all(x in key for x in ["foo", "pixelate", "5"])


def test_get_or_create_gpu_anonymizer_creates_new():
    """GPU anonymizer should be created when configuration changes."""
    config = AnonymizerConfig()
    config.model.name = "foo"
    config.blur.type = BlurType("pixelate")
    config.blur.strength = 5

    with patch("blur_api.serve.Anonymizer") as MockAnonymizer:
        mock_instance = Mock()
        mock_instance.set_runtime_hooks = Mock()
        MockAnonymizer.return_value = mock_instance
        instance = serve.get_or_create_gpu_anonymizer(config, None, None)

        MockAnonymizer.assert_called_once()
        kwargs = MockAnonymizer.call_args.kwargs
        assert kwargs["execution_providers"] is None
        assert instance is mock_instance
        mock_instance.set_runtime_hooks.assert_called_once_with(
            cancel_event=None, progress_callback=None
        )
        assert serve.active_gpu_model_key == serve.get_model_key(config)


def test_get_or_create_gpu_anonymizer_reuses_existing():
    """GPU anonymizer instance should be reused when configuration matches."""
    config = AnonymizerConfig()
    config.model.name = "foo"
    config.blur.type = BlurType("pixelate")
    config.blur.strength = 5
    key = serve.get_model_key(config)

    with patch("blur_api.serve.Anonymizer") as MockAnonymizer:
        mock_instance = Mock()
        mock_instance.set_runtime_hooks = Mock()
        serve.active_gpu_model_key = key
        serve.gpu_anonymizer_instance = mock_instance
        instance = serve.get_or_create_gpu_anonymizer(config, None, None)

        MockAnonymizer.assert_not_called()
        assert instance is mock_instance
        mock_instance.set_runtime_hooks.assert_called_once_with(
            cancel_event=None, progress_callback=None
        )


def test_get_or_create_cpu_anonymizer_creates_new():
    """CPU anonymizer should enforce CPU execution provider."""
    config = AnonymizerConfig()
    config.model.name = "foo"

    with patch("blur_api.serve.Anonymizer") as MockAnonymizer:
        mock_instance = Mock()
        mock_instance.set_runtime_hooks = Mock()
        MockAnonymizer.return_value = mock_instance
        instance = serve.get_or_create_cpu_anonymizer(config, None, None)

        MockAnonymizer.assert_called_once()
        kwargs = MockAnonymizer.call_args.kwargs
        assert kwargs["execution_providers"] == ["CPUExecutionProvider"]
        assert instance is mock_instance
        mock_instance.set_runtime_hooks.assert_called_once_with(
            cancel_event=None, progress_callback=None
        )
        assert serve.active_cpu_model_key == serve.get_model_key(config)


def test_get_or_create_cpu_anonymizer_reuses_existing():
    """CPU anonymizer instance should be reused when keys match."""
    config = AnonymizerConfig()
    key = serve.get_model_key(config)

    with patch("blur_api.serve.Anonymizer") as MockAnonymizer:
        mock_instance = Mock()
        mock_instance.set_runtime_hooks = Mock()
        serve.active_cpu_model_key = key
        serve.cpu_anonymizer_instance = mock_instance

        result = serve.get_or_create_cpu_anonymizer(config, None, None)

        MockAnonymizer.assert_not_called()
        assert result is mock_instance
        mock_instance.set_runtime_hooks.assert_called_once_with(
            cancel_event=None, progress_callback=None
        )


def test_get_or_create_anonymizer_delegates_to_gpu():
    """Legacy get_or_create_anonymizer should delegate to GPU implementation."""
    config = AnonymizerConfig()
    with patch.object(serve, "get_or_create_gpu_anonymizer") as mock_gpu:
        serve.get_or_create_anonymizer(config, None, None)
        mock_gpu.assert_called_once_with(config, None, None)


def test_api_routes():
    """Test API routes are properly configured with correct methods."""
    routes = {route.path: route.methods for route in serve.app.routes}

    assert "/blur/frame" in routes
    assert "POST" in routes["/blur/frame"]
    assert "/blur/image_file" in routes
    assert "POST" in routes["/blur/image_file"]
    assert "/blur/video_file" in routes
    assert "POST" in routes["/blur/video_file"]
    assert "/blur/video_progress/{job_id}" in routes
    assert "GET" in routes["/blur/video_progress/{job_id}"]


def test_blur_frame_success():
    """Test successful frame blurring with valid image data."""
    blurred = np.zeros((10, 10, 3), dtype=np.uint8)

    with (
        patch("blur_api.serve.is_gpu_busy", return_value=False),
        patch("blur_api.serve.get_or_create_gpu_anonymizer") as mock_get_gpu,
        patch("blur_api.serve.get_or_create_cpu_anonymizer") as mock_get_cpu,
    ):
        mock_anon = Mock()
        mock_anon.blur_image_array.return_value = blurred
        mock_get_gpu.return_value = mock_anon

        img = np.ones((10, 10, 3), dtype=np.uint8) * 255
        _, buf = cv2.imencode(".jpg", img)

        response = client.post(
            "/blur/frame", files={"input": ("test.jpg", buf.tobytes(), "image/jpeg")}
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/jpeg")
        mock_get_gpu.assert_called_once()
        mock_get_cpu.assert_not_called()
        mock_anon.blur_image_array.assert_called_once()


def test_blur_frame_falls_back_to_cpu_when_gpu_busy():
    """Frame endpoint should use CPU anonymizer if GPU is busy processing video."""
    blurred = np.zeros((10, 10, 3), dtype=np.uint8)

    with (
        patch("blur_api.serve.is_gpu_busy", return_value=True),
        patch("blur_api.serve.get_or_create_gpu_anonymizer") as mock_get_gpu,
        patch("blur_api.serve.get_or_create_cpu_anonymizer") as mock_get_cpu,
    ):
        mock_cpu_anon = Mock()
        mock_cpu_anon.blur_image_array.return_value = blurred
        mock_get_cpu.return_value = mock_cpu_anon

        img = np.ones((10, 10, 3), dtype=np.uint8) * 255
        _, buf = cv2.imencode(".jpg", img)

        response = client.post(
            "/blur/frame", files={"input": ("test.jpg", buf.tobytes(), "image/jpeg")}
        )

        assert response.status_code == 200
        mock_get_gpu.assert_not_called()
        mock_get_cpu.assert_called_once()
        mock_cpu_anon.blur_image_array.assert_called_once()


def test_blur_frame_invalid_image():
    """Test frame blurring with invalid image data."""
    with patch("blur_api.serve.get_or_create_anonymizer") as mock_get_anon:
        mock_anon = Mock()
        mock_get_anon.return_value = mock_anon
        mock_anon.blur_image_array.side_effect = ValueError("Invalid image data")

        response = client.post(
            "/blur/frame", files={"input": ("test.jpg", b"invalid", "image/jpeg")}
        )
        assert response.status_code == 400
        assert "Invalid image data" in response.text


def test_blur_image_file_success(monkeypatch, tmp_path):
    """Test successful image file blurring."""
    with patch("blur_api.serve.get_or_create_anonymizer") as mock_get_anon:
        mock_anon = Mock()
        mock_get_anon.return_value = mock_anon

        # Setup test files
        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.jpg"
        input_path.write_bytes(b"dummy")
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

        response = client.post(
            "/blur/image_file",
            data={"input": str(input_path), "output_path": str(output_path)},
        )

        assert response.status_code == 200
        assert "message" in response.json()
        mock_anon.blur_image_file.assert_called_once()
        args = mock_anon.blur_image_file.call_args[0]
        assert isinstance(args[0], Path)  # Input path
        assert isinstance(args[1], Path)  # Output path
        assert str(args[0]) == str(input_path)
        assert str(args[1]) == str(output_path)


def test_blur_image_file_not_found(monkeypatch):
    """Test image file blurring with non-existent input file."""
    with patch("blur_api.serve.get_or_create_anonymizer"):
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        response = client.post(
            "/blur/image_file",
            data={"input": "/notfound.jpg", "output_path": "/out.jpg"},
        )
        assert response.status_code == 400
        assert "Input file does not exist" in response.text


def test_blur_video_file_success(monkeypatch, tmp_path):
    """Test successful video file blurring job creation."""
    with patch("blur_api.serve.get_or_create_anonymizer") as mock_get_anon:
        mock_anon = Mock()
        mock_get_anon.return_value = mock_anon

        # Setup test files
        input_path = tmp_path / "input.mp4"
        output_path = tmp_path / "output.mp4"
        input_path.write_bytes(b"dummy")
        monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

        response = client.post(
            "/blur/video_file",
            data={"config": "{}", "output_filename": str(output_path.name)},
            files={"input": ("input.mp4", b"dummy", "video/mp4")},
        )

        assert response.status_code == 200
        result = response.json()
        assert "job_id" in result
        job_id = result["job_id"]
        # Should be a valid UUID
        assert len(job_id) == 36  # UUID format
        assert job_id.count("-") == 4  # UUID format check

        deadline = time.time() + 1.0
        while time.time() < deadline:
            with serve.video_jobs_lock:
                job = serve.video_jobs.get(job_id)
            if not job or job.get("status") != "running":
                break
            time.sleep(0.01)

        with serve.video_jobs_lock:
            serve.video_jobs.pop(job_id, None)
        serve.cancel_events.clear()


def test_video_progress_success():
    """Test video processing progress check with valid job."""
    job_id = "test_job"
    serve.video_jobs[job_id] = {
        "progress": 50,
        "message": "Processing",
        "status": "done",  # Set to done so the stream ends
        "error": None,
    }

    with client.stream("GET", f"/blur/video_progress/{job_id}") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events: list[str] = []
        for line in response.iter_lines():
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if line:
                events.append(line)

        assert events
        event_data = events[0].replace("data: ", "")
        result = json.loads(event_data)
        assert result["progress"] == 50
        assert result["message"] == "Processing"

    serve.video_jobs.pop(job_id, None)


def test_video_progress_not_found():
    """Test video progress check with non-existent job."""
    response = client.get("/blur/video_progress/doesnotexist")
    assert response.status_code == 404
    assert "Job not found" in response.text


def test_is_gpu_busy_reports_status():
    """Verify GPU busy helper reflects lock state."""
    acquired = serve.model_lock.acquire(blocking=False)
    if acquired:
        serve.model_lock.release()

    assert serve.is_gpu_busy() is False

    serve.model_lock.acquire()
    try:
        assert serve.is_gpu_busy() is True
    finally:
        serve.model_lock.release()
