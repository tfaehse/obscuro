import threading
from unittest.mock import Mock

import pytest

from anonymizer.cancellation import CancellationException, CancellationMixin


class DummyCancelable(CancellationMixin):
    def __init__(self, cancel_event=None, progress_callback=None):
        super().__init__(cancel_event=cancel_event, progress_callback=progress_callback)


def test_safe_progress_update_calls_callback():
    callback = Mock()
    cancelable = DummyCancelable(progress_callback=callback)

    cancelable.safe_progress_update(10, "Stage", "Message")

    callback.assert_called_once_with(10, "Stage", "Message")


def test_safe_progress_update_checks_cancellation():
    event = threading.Event()
    event.set()
    cancelable = DummyCancelable(cancel_event=event)

    with pytest.raises(CancellationException):
        cancelable.safe_progress_update(0, "Stage", "Message")
