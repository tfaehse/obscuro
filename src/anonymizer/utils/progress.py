from __future__ import annotations

import time
from collections.abc import Callable

ProgressCallback = Callable[[int, str, str], None]


def throttle_progress_callback(
    callback: ProgressCallback | None,
    *,
    interval_seconds: float = 2.0,
    percent_step: float = 5.0,
) -> ProgressCallback | None:
    """Throttle progress events by time and percentage progression.

    The wrapped callback is emitted immediately on the first call, whenever the
    stage or message changes, upon completion (>=100%), when progress advances by
    at least ``percent_step`` since the last emission, or when ``interval_seconds``
    have elapsed—whichever happens first.
    """
    if callback is None:
        return None

    interval = max(0.0, float(interval_seconds))
    step = max(0.0, float(percent_step))

    last_emit_time = float("-inf")
    last_emit_stage: str | None = None
    last_emit_message: str | None = None
    last_emit_percent: int | None = None

    def wrapped(percent: int, stage: str, message: str) -> None:
        nonlocal last_emit_time, last_emit_stage, last_emit_message, last_emit_percent

        now = time.monotonic()
        first_emit = last_emit_percent is None
        stage_changed = (last_emit_stage is not None) and stage != last_emit_stage
        message_changed = (last_emit_message is not None) and message != last_emit_message
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

        should_emit = first_emit or reached_completion or stage_changed or message_changed
        if not should_emit:
            should_emit = percent_gate_hit or time_gate_hit

        if should_emit:
            callback(percent, stage, message)
            last_emit_time = now
            last_emit_stage = stage
            last_emit_message = message
            last_emit_percent = percent

    return wrapped
