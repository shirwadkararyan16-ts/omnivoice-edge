"""
Benchmark the OmniVoice ONNX forward/logits subgraph with TensorRT.

This script creates an ONNX Runtime session using the TensorRT Execution
Provider with FP16 enabled, runs a warm-up inference, and measures latency
across multiple benchmark iterations.

The benchmark covers only the exported forward/logits subgraph. It does
not measure the complete end-to-end OmniVoice text-to-speech pipeline.
"""

from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "omnivoice_forward.onnx"
)

WEIGHTS_PATH = Path(f"{MODEL_PATH}.data")

CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "tensorrt"
    / "cache"
)

TENSORRT_PROVIDER = "TensorrtExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

BATCH_SIZE = 1
TOKEN_CHANNELS = 8
SEQUENCE_LENGTH = 8
BENCHMARK_RUNS = 20
TENSORRT_WORKSPACE_BYTES = 512 * 1024 * 1024


def create_example_inputs() -> dict[str, np.ndarray]:
    """
    Create representative NumPy inputs for the exported ONNX model.

    Returns:
        A dictionary mapping model input names to NumPy arrays.
    """
    return {
        "input_ids": np.zeros(
            (
                BATCH_SIZE,
                TOKEN_CHANNELS,
                SEQUENCE_LENGTH,
            ),
            dtype=np.int64,
        ),
        "audio_mask": np.zeros(
            (
                BATCH_SIZE,
                SEQUENCE_LENGTH,
            ),
            dtype=np.bool_,
        ),
        "attention_mask": np.ones(
            (
                BATCH_SIZE,
                SEQUENCE_LENGTH,
            ),
            dtype=np.int64,
        ),
        "position_ids": np.arange(
            SEQUENCE_LENGTH,
            dtype=np.int64,
        )[None, :],
    }


def create_session() -> ort.InferenceSession:
    """
    Create an ONNX Runtime session using TensorRT with CUDA fallback.

    Returns:
        An initialized ONNX Runtime inference session.

    Raises:
        RuntimeError: If the TensorRT Execution Provider is unavailable
            or does not become active.
    """
    ort.preload_dlls()

    available_providers = ort.get_available_providers()

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("TensorRT:", trt.__version__)
    print("ONNX Runtime:", ort.__version__)
    print("Available providers:", available_providers)

    if TENSORRT_PROVIDER not in available_providers:
        raise RuntimeError(
            "TensorrtExecutionProvider is unavailable. "
            "Verify that TensorRT and a compatible ONNX Runtime GPU "
            "installation are configured correctly."
        )

    providers = [
        (
            TENSORRT_PROVIDER,
            {
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(
                    CACHE_DIRECTORY.resolve()
                ),
                "trt_timing_cache_enable": True,
                "trt_max_workspace_size": (
                    TENSORRT_WORKSPACE_BYTES
                ),
            },
        ),
        (
            CUDA_PROVIDER,
            {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "do_copy_in_default_stream": True,
            },
        ),
        CPU_PROVIDER,
    ]

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 1

    print()
    print("Creating TensorRT session...")
    print("The first engine build may take several minutes.")

    start_time = time.perf_counter()

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=providers,
    )

    build_seconds = time.perf_counter() - start_time
    active_providers = session.get_providers()

    print()
    print(f"Session created in {build_seconds:.3f} seconds.")
    print("Active providers:", active_providers)
    print("Provider options:", session.get_provider_options())

    if TENSORRT_PROVIDER not in active_providers:
        raise RuntimeError(
            "The session was created, but "
            "TensorrtExecutionProvider is not active."
        )

    return session


def main() -> None:
    """Load the model and benchmark TensorRT FP16 inference latency."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing ONNX model: {MODEL_PATH}"
        )

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing ONNX external weights: {WEIGHTS_PATH}"
        )

    CACHE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    session = create_session()
    inputs = create_example_inputs()

    print()
    print("Running warm-up inference...")

    output = session.run(
        ["logits"],
        inputs,
    )[0]

    print("Warm-up output shape:", output.shape)
    print("Warm-up output dtype:", output.dtype)

    inference_times: list[float] = []

    print()
    print(
        f"Running {BENCHMARK_RUNS} benchmark iterations..."
    )

    for run_number in range(1, BENCHMARK_RUNS + 1):
        start_time = time.perf_counter()

        output = session.run(
            ["logits"],
            inputs,
        )[0]

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        inference_times.append(elapsed_ms)

        print(
            f"Run {run_number}: "
            f"{elapsed_ms:.3f} ms"
        )

    average_ms = float(np.mean(inference_times))
    median_ms = float(np.median(inference_times))
    minimum_ms = float(np.min(inference_times))
    maximum_ms = float(np.max(inference_times))

    print()
    print("TensorRT FP16 benchmark completed.")
    print(f"Average latency: {average_ms:.3f} ms")
    print(f"Median latency: {median_ms:.3f} ms")
    print(f"Minimum latency: {minimum_ms:.3f} ms")
    print(f"Maximum latency: {maximum_ms:.3f} ms")
    print("Output shape:", output.shape)
    print("Output dtype:", output.dtype)
    print("Cache directory:", CACHE_DIRECTORY.resolve())


if __name__ == "__main__":
    main()