"""ONNX Runtime with TensorRT optimization for ultra-fast AI inference."""

__version__ = "1.0.0"
__author__ = "Ümit Kacar"
__email__ = "umitkacar@example.com"
__license__ = "MIT"

from onnxruntime_tensorrt.core.session import TensorRTSession
from onnxruntime_tensorrt.utils.logger import setup_logger

__all__ = [
    "TensorRTSession",
    "__version__",
    "setup_logger",
]
