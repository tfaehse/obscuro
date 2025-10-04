#!/usr/bin/env python3
"""
Logging setup utilities for the blur GUI application.
Provides both standard console logging and optional JSON file logging.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import ClassVar


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class CLIFormatter(logging.Formatter):
    """Custom formatter for CLI output with color support."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors for CLI output."""
        # Don't add colors if output is not a terminal
        if not sys.stdout.isatty():
            return super().format(record)

        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        # Format the message
        formatted = super().format(record)
        return f"{color}{formatted}{reset}"


class ProgressFormatter(logging.Formatter):
    """Special formatter for progress messages that outputs without level prefix."""

    def format(self, record: logging.LogRecord) -> str:
        """Format progress messages simply."""
        return record.getMessage()


def setup_logging(
    log_level: str = "INFO",
    json_log_file: Path | None = None,
    console_format: str = "%(message)s",
    enable_colors: bool = True,
) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_log_file: Optional path to JSON log file
        console_format: Format string for console output
        enable_colors: Whether to enable colored output in terminal

    Returns:
        Configured logger instance
    """
    # Get the root logger for our application
    logger = logging.getLogger("obscuro")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if enable_colors:
        console_formatter = CLIFormatter(console_format)
    else:
        console_formatter = logging.Formatter(console_format)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Error handler for stderr (WARNING and above)
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.WARNING)
    error_formatter = CLIFormatter("ERROR: %(message)s")
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)

    # Special progress logger setup
    progress_logger = logging.getLogger("obscuro.progress")
    progress_logger.setLevel(logging.INFO)
    progress_handler = logging.StreamHandler(sys.stdout)
    progress_handler.setLevel(logging.INFO)
    progress_handler.setFormatter(ProgressFormatter())
    progress_logger.addHandler(progress_handler)
    progress_logger.propagate = False  # Don't propagate to parent

    # JSON file handler (optional)
    if json_log_file:
        json_log_file = Path(json_log_file)
        json_log_file.parent.mkdir(parents=True, exist_ok=True)

        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        json_handler.setLevel(logging.DEBUG)  # Always log everything to JSON
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)

        # Also add JSON handler to progress logger
        progress_logger.addHandler(json_handler)

        logger.info(f"JSON logging enabled: {json_log_file}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_progress_logger() -> logging.Logger:
    """Get a logger specifically for progress updates."""
    progress_logger = logging.getLogger("obscuro.progress")
    return progress_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def log_with_extra(logger: logging.Logger, level: str, message: str, **extra_fields) -> None:
    """
    Log a message with extra fields that will be included in JSON output.

    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        message: Log message
        **extra_fields: Additional fields to include in JSON logs
    """
    # Create a log record with extra fields
    record = logger.makeRecord(
        logger.name, getattr(logging, level.upper()), __file__, 0, message, (), None
    )
    record.extra_fields = extra_fields
    logger.handle(record)
