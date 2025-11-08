"""Tests for logging utilities."""

from __future__ import annotations

import logging

from onnxruntime_tensorrt.utils.logger import setup_logger


class TestLogger:
    """Test suite for logger utilities."""

    def test_setup_logger_default(self) -> None:
        """Test logger setup with default parameters."""
        logger = setup_logger()
        assert logger.name == "onnxruntime_tensorrt"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1

    def test_setup_logger_custom_name(self) -> None:
        """Test logger setup with custom name."""
        logger = setup_logger("custom_logger")
        assert logger.name == "custom_logger"

    def test_setup_logger_custom_level(self) -> None:
        """Test logger setup with custom level."""
        logger = setup_logger(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_setup_logger_custom_format(self) -> None:
        """Test logger setup with custom format string."""
        custom_format = "%(levelname)s - %(message)s"
        logger = setup_logger(format_string=custom_format)
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert handler.formatter is not None
        assert handler.formatter._fmt == custom_format

    def test_logger_clears_existing_handlers(self) -> None:
        """Test that existing handlers are cleared."""
        logger = setup_logger("test_logger")
        initial_handlers = len(logger.handlers)

        # Setup again
        logger = setup_logger("test_logger")
        assert len(logger.handlers) == initial_handlers
