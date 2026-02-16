"""Cleanup orphaned temporary files from crashed processes.

This module provides utilities to clean up temporary directories that were
left behind when the application crashed or was force-killed. It uses
age-based detection - any temp directory older than the threshold is removed.
"""

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger("obscuro.cleanup")

# Default age threshold: 24 hours in seconds
DEFAULT_MAX_AGE_SECONDS = 24 * 60 * 60

# Prefixes for temp directories created by this application
TEMP_PREFIXES = [
    "obscuro_proto_",  # Mask prototype directories (from masks.py)
    "obscuro_jobs",  # API session directories (from serve.py)
]


def cleanup_orphaned_temp_dirs(
    max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
    dry_run: bool = False,
) -> list[Path]:
    """Remove orphaned temporary directories older than the specified age.

    Args:
        max_age_seconds: Maximum age in seconds before a directory is considered
                        orphaned. Defaults to 24 hours.
        dry_run: If True, only report what would be deleted without actually
                removing anything.

    Returns:
        List of paths that were removed (or would be removed in dry_run mode).
    """
    from anonymizer.paths import get_temp_dir

    temp_root = get_temp_dir()
    removed: list[Path] = []
    current_time = time.time()
    cutoff_time = current_time - max_age_seconds

    for prefix in TEMP_PREFIXES:
        # Handle obscuro_jobs specially - it's a parent directory containing sessions
        if prefix == "obscuro_jobs":
            jobs_dir = temp_root / "obscuro_jobs"
            if not jobs_dir.exists():
                continue

            for session_dir in jobs_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                if _should_remove(session_dir, cutoff_time):
                    removed.append(session_dir)
                    if not dry_run:
                        _remove_dir(session_dir)
        else:
            # For other prefixes, scan for matching directories in temp root
            for path in temp_root.iterdir():
                if not path.is_dir():
                    continue
                if not path.name.startswith(prefix):
                    continue
                if _should_remove(path, cutoff_time):
                    removed.append(path)
                    if not dry_run:
                        _remove_dir(path)

    if removed:
        action = "Would remove" if dry_run else "Removed"
        logger.info(
            "%s %d orphaned temp director%s",
            action,
            len(removed),
            "y" if len(removed) == 1 else "ies",
        )
        for path in removed:
            logger.debug("  - %s", path)

    return removed


def _should_remove(path: Path, cutoff_time: float) -> bool:
    """Check if a directory should be removed based on its modification time."""
    try:
        mtime = path.stat().st_mtime
        return mtime < cutoff_time
    except OSError:
        # If we can't stat it, leave it alone
        return False


def _remove_dir(path: Path) -> bool:
    """Remove a directory and all its contents.

    Returns:
        True if removal was successful, False otherwise.
    """
    try:
        shutil.rmtree(path, ignore_errors=True)
        return True
    except OSError as exc:
        logger.warning("Failed to remove orphaned temp dir %s: %s", path, exc)
        return False
