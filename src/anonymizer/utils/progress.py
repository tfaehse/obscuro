from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from datetime import timedelta

ProgressCallback = Callable[[int, str, str], None]


def throttle_progress_callback(
    callback: ProgressCallback | None,
    *,
    interval_seconds: float = 2.0,
    percent_step: float = 5.0,
) -> ProgressCallback | None:
    """Throttle progress events by time and percentage progression.

    The wrapped callback is emitted immediately on the first call, upon
    completion (>=100%), when progress advances by at least ``percent_step``
    since the last emission, or when ``interval_seconds`` have elapsed—
    whichever happens first.
    """
    if callback is None:
        return None

    interval = max(0.0, float(interval_seconds))
    step = max(0.0, float(percent_step))

    last_emit_time = float("-inf")
    last_emit_percent: int | None = None

    def wrapped(percent: int, stage: str, message: str) -> None:
        nonlocal last_emit_time, last_emit_percent

        now = time.monotonic()
        first_emit = last_emit_percent is None
        reached_completion = percent >= 100

        if first_emit:
            percent_gate_hit = True
        elif step <= 0.0:
            percent_gate_hit = percent != last_emit_percent
        elif last_emit_percent is not None:
            percent_gate_hit = (percent - last_emit_percent) >= step
        else:
            percent_gate_hit = False

        time_gate_hit = interval == 0.0 or now - last_emit_time >= interval

        should_emit = first_emit or reached_completion or percent_gate_hit or time_gate_hit

        if should_emit:
            callback(percent, stage, message)
            last_emit_time = now
            last_emit_percent = percent

    return wrapped


class ProgressRateEstimator:
    """Track a rolling average FPS for staged progress reporting."""

    def __init__(self, window: int = 5) -> None:
        self._window = deque(maxlen=max(1, int(window)))

    def record(self, units: int, duration_seconds: float) -> float:
        duration = max(1e-6, float(duration_seconds))
        self._window.append((int(units), duration))
        total_units = sum(units for units, _ in self._window)
        total_time = sum(interval for _, interval in self._window)
        if total_time <= 0:
            return 0.0
        return total_units / total_time


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:
        return "0:00"
    return str(timedelta(seconds=int(seconds)))


def format_progress_message(prefix: str, fps: float, remaining_units: int | None) -> str:
    suffix = f"{fps:.2f}fps"
    if remaining_units is not None and remaining_units >= 0:
        eta_seconds = remaining_units / fps if fps > 0 else 0.0
        suffix += f" | eta {_format_eta(eta_seconds)}"
    return f"{prefix} | {suffix}"
