import os

import pytest

from anonymizer import paths


@pytest.fixture(autouse=True)
def clear_caches():
    paths._default_data_root.cache_clear()
    paths._default_models_dir.cache_clear()
    yield
    paths._default_data_root.cache_clear()
    paths._default_models_dir.cache_clear()


def test_get_data_root_uses_environment_override(tmp_path, monkeypatch):
    override = tmp_path / "data"
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(override))
    result = paths.get_data_root()
    assert result == override
    assert result.exists()


def test_get_models_dir_uses_environment_override(tmp_path, monkeypatch):
    override = tmp_path / "models_override"
    monkeypatch.setenv(paths.ENV_MODELS_DIR, str(override))
    result = paths.get_models_dir()
    assert result == override
    assert result.exists()


def test_ensure_default_model_present_existing(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for filename in paths.DEFAULT_MODEL_FILENAMES.values():
        (models_dir / filename).write_bytes(b"model")

    found = paths.ensure_default_model_present(models_dir=models_dir)
    assert found == models_dir / paths.DEFAULT_MODEL_FILENAME
    for filename in paths.DEFAULT_MODEL_FILENAMES.values():
        assert (models_dir / filename).exists()


def test_ensure_default_model_copies_bundled(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    bundled_dir = tmp_path / "bundled"
    bundled_dir.mkdir()
    for filename in paths.DEFAULT_MODEL_FILENAMES.values():
        (bundled_dir / filename).write_bytes(f"bundled_{filename}".encode())

    monkeypatch.setenv(paths.ENV_MODELS_DIR, str(models_dir))
    monkeypatch.setitem(os.environ, paths.ENV_MODELS_DIR, str(models_dir))

    # Patch resources.files to return our bundled directory
    monkeypatch.setattr(
        paths.resources,
        "files",
        lambda pkg: bundled_dir if pkg == paths._BUNDLED_MODELS_PACKAGE else None,
    )

    paths._default_models_dir.cache_clear()
    target = paths.ensure_default_model_present()
    assert target is not None
    assert target.exists()
    assert target.read_bytes() == b"bundled_" + paths.DEFAULT_MODEL_FILENAME.encode()
    for filename in paths.DEFAULT_MODEL_FILENAMES.values():
        model_path = models_dir / filename
        assert model_path.exists()
        assert model_path.read_bytes() == b"bundled_" + filename.encode()


def test_ensure_default_model_uses_repo_fallback(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo_models = {}
    for filename in paths.DEFAULT_MODEL_FILENAMES.values():
        path = repo_dir / filename
        path.write_bytes(f"repo_{filename}".encode())
        repo_models[filename] = path

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    monkeypatch.setattr(paths, "_get_bundled_model", lambda filename: None)
    monkeypatch.setattr(paths, "_get_repo_model", lambda filename: repo_models.get(filename))

    result = paths.ensure_default_model_present(models_dir=models_dir)
    assert result is not None
    assert result.exists()
    for filename in paths.DEFAULT_MODEL_FILENAMES.values():
        expected = repo_models[filename].read_bytes()
        assert (models_dir / filename).read_bytes() == expected
