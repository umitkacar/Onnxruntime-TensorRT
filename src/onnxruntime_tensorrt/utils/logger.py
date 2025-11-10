"""Logging utilities for ONNX Runtime TensorRT."""

from __future__ import annotations

import logging
import sys


def setup_logger(
    name: str = "onnxruntime_tensorrt",
    level: int = logging.INFO,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        format_string: Custom format string (default: standard format)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("my_app", level=logging.DEBUG)
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger
