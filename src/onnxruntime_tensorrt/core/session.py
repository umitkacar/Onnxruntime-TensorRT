"""TensorRT-optimized ONNX Runtime session management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


class TensorRTSession:
    """
    ONNX Runtime session with TensorRT optimization.

    This class provides a high-level interface for running inference with
    TensorRT-optimized ONNX models.

    Args:
        model_path: Path to the ONNX model file
        use_tensorrt: Enable TensorRT execution provider
        use_cuda: Enable CUDA execution provider
        fp16: Enable FP16 precision mode
        int8: Enable INT8 quantization
        workspace_size: Maximum workspace size for TensorRT (in GB)
        cache_dir: Directory for TensorRT engine cache
        device_id: CUDA device ID

    Example:
        >>> session = TensorRTSession(
        ...     model_path="model.onnx",
        ...     use_tensorrt=True,
        ...     fp16=True,
        ...     workspace_size=4
        ... )
        >>> output = session.run(input_data)
    """

    def __init__(
        self,
        model_path: str | Path,
        use_tensorrt: bool = True,
        use_cuda: bool = True,
        fp16: bool = False,
        int8: bool = False,
        workspace_size: int = 4,
        cache_dir: str | Path | None = None,
        device_id: int = 0,
    ) -> None:
        """Initialize TensorRT session."""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Setup execution providers
        providers = self._setup_providers(
            use_tensorrt=use_tensorrt,
            use_cuda=use_cuda,
            fp16=fp16,
            int8=int8,
            workspace_size=workspace_size,
            cache_dir=cache_dir,
            device_id=device_id,
        )

        # Create inference session
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)

        # Get model metadata
        self._input_names = [inp.name for inp in self.session.get_inputs()]
        self._output_names = [out.name for out in self.session.get_outputs()]
        self._input_shapes = [inp.shape for inp in self.session.get_inputs()]
        self._output_shapes = [out.shape for out in self.session.get_outputs()]

    def _setup_providers(
        self,
        use_tensorrt: bool,
        use_cuda: bool,
        fp16: bool,
        int8: bool,
        workspace_size: int,
        cache_dir: str | Path | None,
        device_id: int,
    ) -> list[str | tuple[str, dict[str, Any]]]:
        """Setup execution providers with configuration."""
        providers: list[str | tuple[str, dict[str, Any]]] = []

        if use_tensorrt:
            trt_options: dict[str, Any] = {
                "device_id": device_id,
                "trt_max_workspace_size": workspace_size * 1024 * 1024 * 1024,
                "trt_fp16_enable": fp16,
                "trt_int8_enable": int8,
                "trt_engine_cache_enable": cache_dir is not None,
            }

            if cache_dir is not None:
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                trt_options["trt_engine_cache_path"] = str(cache_path)

            providers.append(("TensorrtExecutionProvider", trt_options))

        if use_cuda:
            providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")

        return providers

    def run(
        self,
        inputs: np.ndarray | dict[str, np.ndarray],
        output_names: list[str] | None = None,
    ) -> list[np.ndarray] | np.ndarray:
        """
        Run inference on input data.

        Args:
            inputs: Input data as numpy array or dict of {name: array}
            output_names: List of output names to return (default: all)

        Returns:
            Model outputs as list of arrays or single array

        Raises:
            ValueError: If input format is invalid
        """
        # Prepare inputs
        if isinstance(inputs, np.ndarray):
            if len(self._input_names) != 1:
                raise ValueError(
                    f"Model expects {len(self._input_names)} inputs, "
                    f"but single array provided. Use dict input."
                )
            input_dict = {self._input_names[0]: inputs}
        else:
            input_dict = inputs

        # Validate inputs
        for name in input_dict:
            if name not in self._input_names:
                raise ValueError(f"Unknown input name: {name}")

        # Run inference
        outputs = self.session.run(output_names, input_dict)

        # Return single output if only one
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    @property
    def input_names(self) -> list[str]:
        """Get input tensor names."""
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Get output tensor names."""
        return self._output_names

    @property
    def input_shapes(self) -> list[list[int | str]]:
        """Get input tensor shapes."""
        return self._input_shapes

    @property
    def output_shapes(self) -> list[list[int | str]]:
        """Get output tensor shapes."""
        return self._output_shapes

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TensorRTSession("
            f"model={self.model_path.name}, "
            f"inputs={len(self._input_names)}, "
            f"outputs={len(self._output_names)})"
        )
