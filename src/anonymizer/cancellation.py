"""Cancellation handling utilities for the anonymizer."""


class CancellationException(Exception):
    """Exception raised when operation is cancelled."""

    pass


class CancellationMixin:
    """Mixin to handle cancellation cleanly."""

    def __init__(self, *args, cancel_event=None, progress_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancel_event = cancel_event
        self.progress_callback = progress_callback

    def check_cancellation(self):
        """Check if operation should be cancelled and raise exception if so."""
        if self.cancel_event and self.cancel_event.is_set():
            raise CancellationException("Operation was cancelled")

    def safe_progress_update(self, percent: float, stage: str, message: str):
        """Update progress and check for cancellation."""
        self.check_cancellation()
        if self.progress_callback:
            self.progress_callback(percent, stage, message)
