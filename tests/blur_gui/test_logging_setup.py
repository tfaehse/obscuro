"""
Tests for the logging setup functionality.
"""

import json
import logging
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from blur_cli.logging_setup import (
    CLIFormatter,
    JSONFormatter,
    ProgressFormatter,
    get_logger,
    get_progress_logger,
    log_with_extra,
    setup_logging,
)


class TestLoggingSetup:
    """Test logging setup functionality."""

    def test_setup_logging_info_level(self):
        """Test setting up logging at INFO level."""
        logger = setup_logging(log_level="INFO")

        # Check that the returned logger is configured correctly
        assert logger.level == logging.INFO
        assert logger.name == "obscuro"

    def test_setup_logging_debug_level(self):
        """Test setting up logging at DEBUG level."""
        logger = setup_logging(log_level="DEBUG")

        # Check that the returned logger is configured for debug
        assert logger.level == logging.DEBUG
        assert logger.name == "obscuro"

    def test_setup_logging_warning_level(self):
        """Test setting up logging at WARNING level."""
        logger = setup_logging(log_level="WARNING")

        # Check that logger is configured for warning
        assert logger.level == logging.WARNING

    def test_setup_logging_error_level(self):
        """Test setting up logging at ERROR level."""
        logger = setup_logging(log_level="ERROR")

        # Check that logger is configured for error
        assert logger.level == logging.ERROR

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_different_names(self):
        """Test getting loggers with different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 is not logger2

    def test_get_logger_same_name_returns_same_instance(self):
        """Test that getting logger with same name returns same instance."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")

        assert logger1 is logger2

    def test_logging_format_configuration(self):
        """Test that logging format is properly configured."""
        logger = setup_logging(log_level="DEBUG")

        # This should not raise an exception
        logger.info("Test message")
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_setup_logging_with_invalid_level(self):
        """Test setup_logging handles invalid log levels."""
        with pytest.raises(AttributeError):
            setup_logging(log_level="INVALID_LEVEL")

    def test_logger_hierarchy(self):
        """Test that loggers follow proper hierarchy."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")

        assert child_logger.parent == parent_logger

    def test_multiple_setup_calls(self):
        """Test that multiple setup_logging calls work correctly."""
        logger1 = setup_logging(log_level="INFO")
        logger2 = setup_logging(log_level="DEBUG")

        # Should return the same logger instance but with updated level
        assert logger1.name == logger2.name == "obscuro"
        assert logger2.level == logging.DEBUG

    def test_logging_output_destination(self):
        """Test that logs go to expected destinations."""
        logger = setup_logging(log_level="DEBUG")

        # Check that logger has handlers
        assert len(logger.handlers) > 0

        # Should have console and error handlers
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types

    def test_logging_with_special_characters(self):
        """Test logging with special characters and unicode."""
        logger = setup_logging(log_level="DEBUG")

        # These should not raise exceptions
        logger.info("Message with émojis 🔥 and ñ special chars")
        logger.debug("Message with\nnewlines\tand\ttabs")

    def test_concurrent_logging_setup(self):
        """Test that logging setup is thread-safe."""
        import threading

        results = []

        def setup_in_thread():
            logger = setup_logging(log_level="INFO")
            results.append(logger)

        threads = [threading.Thread(target=setup_in_thread) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All should return the same logger instance (by name)
        assert all(logger.name == "obscuro" for logger in results)

    def test_get_progress_logger(self):
        """Test getting progress logger."""
        progress_logger = get_progress_logger()

        assert isinstance(progress_logger, logging.Logger)
        assert progress_logger.name == "obscuro.progress"

    def test_logging_propagation(self):
        """Test logger propagation settings."""
        logger = setup_logging(log_level="INFO")

        # obscuro logger should not propagate to root
        assert logger.propagate is False

    def test_json_logging_option(self):
        """Test JSON logging file option."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_file = Path(tmp_dir) / "test.log"
            logger = setup_logging(log_level="DEBUG", json_log_file=json_file)

            logger.info("Test message")

            # JSON file should be created
            assert json_file.exists()

            # Close handlers targeting the temporary file so Windows can clean up.
            progress_logger = get_progress_logger()
            for target_logger in (logger, progress_logger):
                for handler in target_logger.handlers[:]:
                    if getattr(handler, "baseFilename", None) == str(json_file):
                        handler.close()
                        target_logger.removeHandler(handler)

    def test_color_formatting_option(self):
        """Test color formatting enable/disable."""
        # Test with colors enabled
        logger1 = setup_logging(log_level="INFO", enable_colors=True)
        assert logger1 is not None

        # Test with colors disabled
        logger2 = setup_logging(log_level="INFO", enable_colors=False)
        assert logger2 is not None


class TestJSONFormatter:
    """Test the JSONFormatter class."""

    def test_json_formatter_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test_logger"
        assert parsed["message"] == "Test message"
        assert parsed["line"] == 42
        assert "timestamp" in parsed

    def test_json_formatter_with_exception(self):
        """Test JSON formatting with exception info."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "Error occurred"
        assert "exception" in parsed
        assert "Test exception" in parsed["exception"]

    def test_json_formatter_with_extra_fields(self):
        """Test JSON formatting with extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Add extra fields to the record
        record.extra_fields = {"user_id": 123, "action": "login"}

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["user_id"] == 123
        assert parsed["action"] == "login"


class TestCLIFormatter:
    """Test the CLIFormatter class."""

    def test_cli_formatter_with_colors(self):
        """Test CLI formatting with colors when output is a terminal."""
        formatter = CLIFormatter("%(levelname)s: %(message)s")

        with patch("sys.stdout.isatty", return_value=True):
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Test error",
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)

            # Should contain ANSI color codes for red (ERROR)
            assert "\033[31m" in result  # Red color code
            assert "\033[0m" in result  # Reset code
            assert "ERROR: Test error" in result

    def test_cli_formatter_without_colors(self):
        """Test CLI formatting without colors when output is not a terminal."""
        formatter = CLIFormatter("%(levelname)s: %(message)s")

        with patch("sys.stdout.isatty", return_value=False):
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Test error",
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)

            # Should NOT contain ANSI color codes
            assert "\033[31m" not in result
            assert "\033[0m" not in result
            assert result == "ERROR: Test error"

    def test_cli_formatter_unknown_level(self):
        """Test CLI formatting with unknown log level."""
        formatter = CLIFormatter("%(levelname)s: %(message)s")

        with patch("sys.stdout.isatty", return_value=True):
            record = logging.LogRecord(
                name="test_logger",
                level=35,  # Custom level
                pathname="test.py",
                lineno=42,
                msg="Custom message",
                args=(),
                exc_info=None,
            )
            record.levelname = "CUSTOM"

            result = formatter.format(record)

            # Should still work but without specific color
            assert "CUSTOM: Custom message" in result
            assert "\033[0m" in result  # Reset code should still be there

    def test_cli_formatter_all_levels(self):
        """Test CLI formatting with all standard log levels."""
        formatter = CLIFormatter("%(levelname)s: %(message)s")

        levels = [
            (logging.DEBUG, "DEBUG", "\033[36m"),  # Cyan
            (logging.INFO, "INFO", "\033[32m"),  # Green
            (logging.WARNING, "WARNING", "\033[33m"),  # Yellow
            (logging.ERROR, "ERROR", "\033[31m"),  # Red
            (logging.CRITICAL, "CRITICAL", "\033[35m"),  # Magenta
        ]

        with patch("sys.stdout.isatty", return_value=True):
            for level_num, level_name, expected_color in levels:
                record = logging.LogRecord(
                    name="test_logger",
                    level=level_num,
                    pathname="test.py",
                    lineno=42,
                    msg=f"Test {level_name} message",
                    args=(),
                    exc_info=None,
                )

                result = formatter.format(record)

                assert expected_color in result
                assert "\033[0m" in result  # Reset code
                assert f"{level_name}: Test {level_name} message" in result


class TestProgressFormatter:
    """Test the ProgressFormatter class."""

    def test_progress_formatter(self):
        """Test progress message formatting."""
        formatter = ProgressFormatter()
        record = logging.LogRecord(
            name="progress_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Processing file %s (%d/%d)",
            args=("test.jpg", 5, 10),
            exc_info=None,
        )

        result = formatter.format(record)

        # Should only contain the message, no level or other info
        assert result == "Processing file test.jpg (5/10)"


class TestLogWithExtra:
    """Test the log_with_extra function."""

    def test_log_with_extra_function(self):
        """Test logging with extra fields."""
        # Set up a logger with JSON formatter to capture extra fields
        logger = logging.getLogger("test_extra")
        logger.setLevel(logging.INFO)

        # Capture the log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        try:
            # Use the log_with_extra function
            log_with_extra(
                logger,
                "info",
                "User action logged",
                user_id=456,
                action="download",
                filename="document.pdf",
            )

            # Get the output and parse as JSON
            output = stream.getvalue().strip()
            parsed = json.loads(output)

            assert parsed["message"] == "User action logged"
            assert parsed["user_id"] == 456
            assert parsed["action"] == "download"
            assert parsed["filename"] == "document.pdf"

        finally:
            logger.removeHandler(handler)

    def test_log_with_extra_different_levels(self):
        """Test log_with_extra with different log levels."""
        logger = logging.getLogger("test_extra_levels")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        try:
            levels = ["debug", "info", "warning", "error", "critical"]

            for level in levels:
                stream.seek(0)
                stream.truncate(0)

                log_with_extra(logger, level, f"Test {level} message", test_level=level)

                output = stream.getvalue().strip()
                parsed = json.loads(output)

                assert parsed["level"] == level.upper()
                assert parsed["message"] == f"Test {level} message"
                assert parsed["test_level"] == level

        finally:
            logger.removeHandler(handler)
