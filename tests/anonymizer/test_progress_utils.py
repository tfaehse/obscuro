from anonymizer.utils.progress import throttle_progress_callback


def test_throttle_emits_on_percent_step_and_time(monkeypatch):
    calls: list[tuple[int, str, str]] = []
    clock = {"value": 0.0}

    def fake_monotonic() -> float:
        return clock["value"]

    monkeypatch.setattr("anonymizer.utils.progress.time.monotonic", fake_monotonic)

    cb = throttle_progress_callback(lambda p, s, m: calls.append((p, s, m)))
    assert cb is not None

    cb(0, "stage", "msg")
    assert calls == [(0, "stage", "msg")]

    cb(2, "stage", "msg")
    assert len(calls) == 1

    cb(5, "stage", "msg")
    assert calls[-1] == (5, "stage", "msg")
    assert len(calls) == 2

    clock["value"] += 2.1
    cb(6, "stage", "msg")
    assert calls[-1] == (6, "stage", "msg")
    assert len(calls) == 3

    cb(6, "stage", "msg2")
    assert calls[-1] == (6, "stage", "msg")
    assert len(calls) == 3

    cb(6, "stage-2", "msg2")
    assert calls[-1] == (6, "stage", "msg")
    assert len(calls) == 3

    clock["value"] += 2.1
    cb(6, "stage-2", "msg2")
    assert calls[-1] == (6, "stage-2", "msg2")
    assert len(calls) == 4

    cb(100, "done", "finished")
    assert calls[-1] == (100, "done", "finished")
    assert len(calls) == 5
