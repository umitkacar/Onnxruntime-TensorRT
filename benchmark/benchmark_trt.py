"""
Comprehensive Benchmark Suite for ONNX Runtime + TensorRT
Compare performance across different backends and configurations
"""

import argparse
from dataclasses import asdict, dataclass
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort


@dataclass
class BenchmarkResult:
    """Store benchmark results"""

    backend: str
    precision: str
    batch_size: int
    input_shape: Tuple[int, int]
    avg_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_fps: float
    memory_mb: float


class ModelBenchmark:
    """
    Benchmark ONNX models with different configurations

    Backends:
    - CPU
    - CUDA
    - TensorRT FP32
    - TensorRT FP16
    - TensorRT INT8 (if available)
    """

    def __init__(self, model_path: str, warmup_runs: int = 10, test_runs: int = 100):
        self.model_path = model_path
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.results: List[BenchmarkResult] = []

    def create_session(self, backend: str, precision: str = "FP32") -> ort.InferenceSession:
        """Create ONNX Runtime session with specified backend"""

        providers = []

        if backend == "tensorrt":
            trt_options = {
                "device_id": 0,
                "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_benchmark_cache",
            }

            if precision == "FP16":
                trt_options["trt_fp16_enable"] = True
            elif precision == "INT8":
                trt_options["trt_int8_enable"] = True

            providers.append(("TensorrtExecutionProvider", trt_options))
            providers.append("CUDAExecutionProvider")

        elif backend == "cuda":
            providers.append("CUDAExecutionProvider")

        elif backend == "cpu":
            providers.append("CPUExecutionProvider")

        else:
            raise ValueError(f"Unknown backend: {backend}")

        return ort.InferenceSession(self.model_path, providers=providers)

    def benchmark_session(
        self,
        session: ort.InferenceSession,
        input_data: Dict[str, np.ndarray],
        backend: str,
        precision: str,
        batch_size: int,
        input_shape: Tuple[int, int],
    ) -> BenchmarkResult:
        """Run benchmark on a session"""

        print(f"\nBenchmarking: {backend} - {precision} - Batch {batch_size} - Shape {input_shape}")

        # Warmup
        print(f"Warming up ({self.warmup_runs} runs)...")
        for _ in range(self.warmup_runs):
            session.run(None, input_data)

        # Benchmark
        print(f"Running benchmark ({self.test_runs} runs)...")
        latencies = []

        for i in range(self.test_runs):
            start_time = time.perf_counter()
            session.run(None, input_data)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            if (i + 1) % 20 == 0:
                avg_so_far = np.mean(latencies)
                print(f"  Progress: {i+1}/{self.test_runs} | Avg: {avg_so_far:.2f}ms")

        # Calculate statistics
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        throughput = (1000 / avg_latency) * batch_size

        print("Results:")
        print(f"  Average: {avg_latency:.2f} ± {std_latency:.2f} ms")
        print(f"  Min/Max: {min_latency:.2f} / {max_latency:.2f} ms")
        print(f"  Throughput: {throughput:.2f} samples/sec")

        return BenchmarkResult(
            backend=backend,
            precision=precision,
            batch_size=batch_size,
            input_shape=input_shape,
            avg_latency_ms=float(avg_latency),
            std_latency_ms=float(std_latency),
            min_latency_ms=float(min_latency),
            max_latency_ms=float(max_latency),
            throughput_fps=float(throughput),
            memory_mb=0.0,  # TODO: Implement memory tracking
        )

    def run_comprehensive_benchmark(
        self,
        batch_sizes: Optional[List[int]] = None,
        input_shapes: Optional[List[Tuple[int, int]]] = None,
        backends: Optional[List[str]] = None,
        precisions: Optional[Dict[str, List[str]]] = None,
    ):
        """Run comprehensive benchmark across all configurations"""

        if backends is None:
            backends = ["cpu", "cuda", "tensorrt"]
        if input_shapes is None:
            input_shapes = [(640, 640)]
        if batch_sizes is None:
            batch_sizes = [1, 4, 8]
        if precisions is None:
            precisions = {"cpu": ["FP32"], "cuda": ["FP32"], "tensorrt": ["FP32", "FP16"]}

        # Get model input info
        temp_session = ort.InferenceSession(self.model_path)
        input_info = temp_session.get_inputs()[0]
        input_name = input_info.name

        print(f"\n{'='*60}")
        print(f"Model: {self.model_path}")
        print(f"Input: {input_name} - {input_info.shape}")
        print(f"{'='*60}")

        # Run benchmarks
        for backend in backends:
            for precision in precisions.get(backend, ["FP32"]):
                for batch_size in batch_sizes:
                    for input_shape in input_shapes:
                        try:
                            # Create dummy input
                            input_data = {
                                input_name: np.random.randn(
                                    batch_size, 3, input_shape[0], input_shape[1]
                                ).astype(np.float32)
                            }

                            # Create session
                            session = self.create_session(backend, precision)

                            # Run benchmark
                            result = self.benchmark_session(
                                session, input_data, backend, precision, batch_size, input_shape
                            )

                            self.results.append(result)

                        except Exception as e:
                            print(f"Failed: {backend} - {precision} - Batch {batch_size}")
                            print(f"Error: {e!s}")
                            continue

    def print_results_table(self):
        """Print results in a formatted table"""

        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS")
        print("=" * 100)
        print(
            f"{'Backend':<15} {'Precision':<10} {'Batch':<8} {'Shape':<12} "
            f"{'Avg (ms)':<12} {'Std (ms)':<12} {'Throughput (fps)':<20}"
        )
        print("-" * 100)

        for result in sorted(self.results, key=lambda x: x.avg_latency_ms):
            shape_str = f"{result.input_shape[0]}x{result.input_shape[1]}"
            print(
                f"{result.backend:<15} {result.precision:<10} "
                f"{result.batch_size:<8} {shape_str:<12} "
                f"{result.avg_latency_ms:<12.2f} {result.std_latency_ms:<12.2f} "
                f"{result.throughput_fps:<20.2f}"
            )

        print("=" * 100)

    def calculate_speedups(self):
        """Calculate speedup compared to CPU baseline"""

        # Find CPU FP32 baseline for each configuration
        print("\n" + "=" * 80)
        print("SPEEDUP ANALYSIS (vs CPU FP32 Baseline)")
        print("=" * 80)

        for batch_size in {r.batch_size for r in self.results}:
            for input_shape in {r.input_shape for r in self.results}:
                # Find baseline
                baseline = next(
                    (
                        r
                        for r in self.results
                        if r.backend == "cpu"
                        and r.precision == "FP32"
                        and r.batch_size == batch_size
                        and r.input_shape == input_shape
                    ),
                    None,
                )

                if baseline is None:
                    continue

                print(f"\nBatch: {batch_size} | Shape: {input_shape[0]}x{input_shape[1]}")
                print(f"{'Backend':<15} {'Precision':<10} {'Latency (ms)':<15} {'Speedup':<10}")
                print("-" * 50)

                for result in self.results:
                    if result.batch_size == batch_size and result.input_shape == input_shape:
                        speedup = baseline.avg_latency_ms / result.avg_latency_ms
                        print(
                            f"{result.backend:<15} {result.precision:<10} "
                            f"{result.avg_latency_ms:<15.2f} {speedup:<10.2f}x"
                        )

        print("=" * 80)

    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save results to JSON file"""

        results_dict = [asdict(r) for r in self.results]

        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX models with TensorRT")
    parser.add_argument("model", type=str, help="Path to ONNX model")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=100, help="Test runs")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8], help="Batch sizes to test"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json", help="Output JSON file"
    )

    args = parser.parse_args()

    # Create benchmark
    benchmark = ModelBenchmark(model_path=args.model, warmup_runs=args.warmup, test_runs=args.runs)

    # Run comprehensive benchmark
    benchmark.run_comprehensive_benchmark(
        batch_sizes=args.batch_sizes,
        input_shapes=[(640, 640)],
        backends=["cpu", "cuda", "tensorrt"],
        precisions={"cpu": ["FP32"], "cuda": ["FP32"], "tensorrt": ["FP32", "FP16"]},
    )

    # Print results
    benchmark.print_results_table()
    benchmark.calculate_speedups()

    # Save results
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
