import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from anonymizer.config import AnonymizerConfig
from blur_api import serve


@pytest.fixture(autouse=True)
def temp_models_dir(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    monkeypatch.setattr(serve, "MODELS_DIR", models_dir, raising=False)

    # Ensure helper functions see new directory
    yield

    serve.MODELS_DIR = models_dir


def test_build_effective_config_invalid_json():
    base = AnonymizerConfig()
    with pytest.raises(serve.HTTPException) as exc:
        serve.build_effective_config("not-json", base)
    assert exc.value.status_code == 400


def test_build_effective_config_unknown_key():
    base = AnonymizerConfig()
    with pytest.raises(serve.HTTPException) as exc:
        serve.build_effective_config(json.dumps({"unknown": 1}), base)
    assert exc.value.status_code == 400


def test_build_effective_config_overrides():
    base = AnonymizerConfig()
    config = serve.build_effective_config(json.dumps({"blur": {"strength": 20}}), base)
    assert config.blur.strength == 20


def test_make_progress_callback_updates_job(monkeypatch):
    job_id = "job"
    serve.video_jobs[job_id] = {"progress": 0, "stage": "", "stage_message": "", "message": ""}
    cb = serve.make_progress_callback(job_id)
    cb(50, "Stage", "Message")
    assert serve.video_jobs[job_id]["progress"] == 50
    assert serve.video_jobs[job_id]["stage_message"] == "Message"
    serve.video_jobs.pop(job_id, None)


@pytest.mark.asyncio
async def test_cancel_video_processing_success(monkeypatch):
    job_id = "job"
    serve.video_jobs[job_id] = {"status": "running"}
    event = Mock()
    serve.cancel_events[job_id] = event

    response = await serve.cancel_video_processing(job_id)
    assert response["message"] == "Job cancelled successfully"
    assert serve.video_jobs[job_id]["status"] == "cancelled"
    event.set.assert_called_once()

    serve.video_jobs.clear()
    serve.cancel_events.clear()


def test_model_endpoints(tmp_path, monkeypatch):
    client = TestClient(serve.app)

    # Initially no models
    res = client.get("/blur/models")
    assert res.status_code == 200
    assert res.json() == {"models": []}

    # Upload a model
    model_bytes = b"fake-onnx"
    res = client.post(
        "/blur/models",
        files={"file": ("custom.onnx", model_bytes, "application/octet-stream")},
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["models"]) == 1
    model_entry = data["models"][0]
    filename = model_entry["filename"]
    assert "modified_at" in model_entry
    assert model_entry["size_bytes"] == len(model_bytes)

    # Config options should surface the model list with metadata
    res = client.get("/blur/config/options")
    assert res.status_code == 200
    options = res.json()
    assert model_entry["name"] in options["model"]["available"]
    assert options["model"].get("files")
    assert any("modified_at" in info for info in options["model"]["files"])
    assert "current_single_pass" in options["detection"]

    # Delete the model
    res = client.delete(f"/blur/models/{filename}")
    assert res.status_code == 200
    assert res.json() == {"models": []}
