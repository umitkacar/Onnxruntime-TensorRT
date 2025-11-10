"""Tests for package version and metadata."""

from __future__ import annotations

import re

import onnxruntime_tensorrt


class TestVersion:
    """Test suite for package metadata."""

    def test_version_exists(self) -> None:
        """Test that version attribute exists."""
        assert hasattr(onnxruntime_tensorrt, "__version__")

    def test_version_format(self) -> None:
        """Test that version follows semantic versioning."""
        version = onnxruntime_tensorrt.__version__
        # Semantic versioning pattern: MAJOR.MINOR.PATCH
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"
        assert re.match(pattern, version), f"Invalid version format: {version}"

    def test_author_exists(self) -> None:
        """Test that author attribute exists."""
        assert hasattr(onnxruntime_tensorrt, "__author__")
        assert isinstance(onnxruntime_tensorrt.__author__, str)

    def test_license_exists(self) -> None:
        """Test that license attribute exists."""
        assert hasattr(onnxruntime_tensorrt, "__license__")
        assert onnxruntime_tensorrt.__license__ == "MIT"

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        assert hasattr(onnxruntime_tensorrt, "__all__")
        expected_exports = ["TensorRTSession", "setup_logger", "__version__"]
        for export in expected_exports:
            assert export in onnxruntime_tensorrt.__all__
