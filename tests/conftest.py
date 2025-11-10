"""Pytest configuration and fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_input() -> np.ndarray:
    """Create sample input data for testing."""
    return np.random.randn(1, 3, 224, 224).astype(np.float32)


@pytest.fixture
def sample_batch_input() -> np.ndarray:
    """Create sample batch input data."""
    return np.random.randn(4, 3, 224, 224).astype(np.float32)


@pytest.fixture
def mock_model_path(tmp_path: Path) -> Path:
    """Create a mock model path."""
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    return model_path


@pytest.fixture
def trt_config() -> dict[str, Any]:
    """TensorRT configuration for testing."""
    return {
        "use_tensorrt": False,  # Disable for unit tests
        "use_cuda": False,
        "fp16": False,
        "workspace_size": 1,
    }


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory."""
    cache_dir = tmp_path / "trt_cache"
    cache_dir.mkdir()
    return cache_dir
