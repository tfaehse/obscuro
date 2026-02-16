"""Tests for tempfile_cleanup module."""

import time
from pathlib import Path

import pytest

from anonymizer.paths import get_temp_dir
from anonymizer.tempfile_cleanup import cleanup_orphaned_temp_dirs


@pytest.fixture
def temp_dir_with_orphans(tmp_path: Path, monkeypatch):
    """Create a temp directory structure with orphaned and fresh directories."""
    # Clear cache and monkeypatch get_temp_dir to return our test directory
    get_temp_dir.cache_clear()
    monkeypatch.setattr(
        "anonymizer.paths.get_temp_dir",
        lambda: tmp_path,
    )

    # Create a directory that looks like an old obscuro_proto directory
    old_proto = tmp_path / "obscuro_proto_abc123"
    old_proto.mkdir()

    # Create a directory that looks like a fresh obscuro_proto directory
    fresh_proto = tmp_path / "obscuro_proto_xyz789"
    fresh_proto.mkdir()

    # Create obscuro_jobs directory with old and fresh sessions
    jobs_dir = tmp_path / "obscuro_jobs"
    jobs_dir.mkdir()

    old_session = jobs_dir / "session_old123"
    old_session.mkdir()

    fresh_session = jobs_dir / "session_fresh456"
    fresh_session.mkdir()

    # Create a directory that doesn't match our prefixes (should be ignored)
    unrelated = tmp_path / "some_other_dir"
    unrelated.mkdir()

    # Set modification times: old directories get a time in the past
    old_time = time.time() - (25 * 60 * 60)  # 25 hours ago

    # Note: We need to set both mtime and atime for the directory itself
    import os

    os.utime(old_proto, (old_time, old_time))
    os.utime(old_session, (old_time, old_time))

    return {
        "tmp_path": tmp_path,
        "old_proto": old_proto,
        "fresh_proto": fresh_proto,
        "old_session": old_session,
        "fresh_session": fresh_session,
        "unrelated": unrelated,
    }


def test_cleanup_removes_old_directories(temp_dir_with_orphans):
    """Test that directories older than 24 hours are removed."""
    # Fixture has already monkeypatched get_temp_dir()

    removed = cleanup_orphaned_temp_dirs(dry_run=False)

    # Old directories should be removed
    assert temp_dir_with_orphans["old_proto"] in removed
    assert temp_dir_with_orphans["old_session"] in removed

    # Fresh directories should still exist
    assert temp_dir_with_orphans["fresh_proto"].exists()
    assert temp_dir_with_orphans["fresh_session"].exists()

    # Unrelated directories should be untouched
    assert temp_dir_with_orphans["unrelated"].exists()


def test_cleanup_dry_run(temp_dir_with_orphans):
    """Test that dry_run doesn't actually remove directories."""
    # Fixture has already monkeypatched get_temp_dir()

    removed = cleanup_orphaned_temp_dirs(dry_run=True)

    # Should report what would be removed
    assert temp_dir_with_orphans["old_proto"] in removed
    assert temp_dir_with_orphans["old_session"] in removed

    # But directories should still exist
    assert temp_dir_with_orphans["old_proto"].exists()
    assert temp_dir_with_orphans["old_session"].exists()


def test_cleanup_with_custom_max_age(temp_dir_with_orphans):
    """Test that custom max_age_seconds is respected."""
    # Fixture has already monkeypatched get_temp_dir()

    # With max age of 1 hour, the 25-hour-old directories should be removed
    removed = cleanup_orphaned_temp_dirs(max_age_seconds=1 * 60 * 60, dry_run=False)

    assert temp_dir_with_orphans["old_proto"] in removed
    assert temp_dir_with_orphans["old_session"] in removed
    assert not temp_dir_with_orphans["old_proto"].exists()
    assert not temp_dir_with_orphans["old_session"].exists()


def test_cleanup_empty_temp_dir(monkeypatch, tmp_path):
    """Test that cleanup handles missing directories gracefully."""
    # Clear cache and monkeypatch get_temp_dir
    get_temp_dir.cache_clear()
    monkeypatch.setattr(
        "anonymizer.paths.get_temp_dir",
        lambda: tmp_path,
    )

    # Should not raise even with no matching directories
    removed = cleanup_orphaned_temp_dirs(dry_run=False)
    assert removed == []


def test_cleanup_preserves_non_matching_dirs(temp_dir_with_orphans):
    """Test that directories not matching our prefixes are preserved."""
    # Make the unrelated directory old too
    import os

    old_time = time.time() - (25 * 60 * 60)
    os.utime(temp_dir_with_orphans["unrelated"], (old_time, old_time))

    cleanup_orphaned_temp_dirs(dry_run=False)

    # Unrelated directory should still exist despite being old
    assert temp_dir_with_orphans["unrelated"].exists()
