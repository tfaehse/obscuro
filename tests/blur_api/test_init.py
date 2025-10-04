"""
Tests for blur_api.__init__ module.

This module tests the initialization and server startup functionality
of the blur_api package.
"""

from unittest.mock import patch

from blur_api import start_server


def test_start_server():
    """Test server startup with default configuration."""
    with patch("uvicorn.run") as mock_run:
        start_server()
        mock_run.assert_called_once()
        # Verify default args
        args, kwargs = mock_run.call_args
        assert args[0] is not None  # app instance passed positionally
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 8000
        assert kwargs["log_level"] == "info"
        assert kwargs["reload"] is False


def test_start_server_custom_config():
    """Test server startup with custom host and port."""
    with patch("uvicorn.run") as mock_run:
        start_server(host="0.0.0.0", port=9000)
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9000


def test_start_server_with_reload():
    """Test server startup with reload enabled."""
    with patch("uvicorn.run") as mock_run:
        start_server(reload=True)
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["reload"] is True
