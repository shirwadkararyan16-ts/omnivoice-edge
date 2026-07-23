"""
Benchmark the exported OmniVoice ONNX forward/logits subgraph with CUDA.

This script loads the exported ONNX model using ONNX Runtime's CUDA
Execution Provider, prints model metadata, performs a warm-up inference,
and measures latency across multiple inference runs.

The benchmark covers only the exported forward/logits subgraph, not the
complete OmniVoice text-to-speech pipeline.
"""

from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "omnivoice_forward.onnx"
)
EXTERNAL_DATA_PATH = Path(f"{MODEL_PATH}.data")

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

BATCH_SIZE = 1
TOKEN_CHANNELS = 8
SEQUENCE_LENGTH = 8
BENCHMARK_RUNS = 20


def print_model_information(
    session: ort.InferenceSession,
) -> None:
    """
    Print the names, shapes, and data types of model inputs and outputs.

    Args:
        session: Initialized ONNX Runtime inference session.
    """
    print("\nModel inputs:")

    for model_input in session.get_inputs():
        print(
            f"  {model_input.name}: "
            f"shape={model_input.shape}, "
            f"type={model_input.type}"
        )

    print("\nModel outputs:")

    for model_output in session.get_outputs():
        print(
            f"  {model_output.name}: "
            f"shape={model_output.shape}, "
            f"type={model_output.type}"
        )


def create_example_inputs() -> dict[str, np.ndarray]:
    """
    Create representative NumPy tensors for ONNX inference.

    Returns:
        A dictionary mapping ONNX input names to input tensors.
    """
    input_ids = np.zeros(
        (
            BATCH_SIZE,
            TOKEN_CHANNELS,
            SEQUENCE_LENGTH,
        ),
        dtype=np.int64,
    )

    audio_mask = np.zeros(
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        dtype=np.bool_,
    )

    attention_mask = np.ones(
        (
            BATCH_SIZE,
            SEQUENCE_LENGTH,
        ),
        dtype=np.int64,
    )

    position_ids = np.arange(
        SEQUENCE_LENGTH,
        dtype=np.int64,
    )[None, :]

    return {
        "input_ids": input_ids,
        "audio_mask": audio_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def create_session() -> ort.InferenceSession:
    """
    Create an optimized ONNX Runtime session using CUDA.

    Returns:
        An initialized ONNX Runtime inference session.

    Raises:
        RuntimeError: If the CUDA Execution Provider is unavailable or
            is not activated for the created session.
    """
    available_providers = ort.get_available_providers()

    print("ONNX Runtime version:", ort.__version__)
    print("Available providers:", available_providers)

    if CUDA_PROVIDER not in available_providers:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. "
            "Install a compatible onnxruntime-gpu package and verify "
            "your CUDA environment."
        )

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    providers = [
        (
            CUDA_PROVIDER,
            {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
            },
        ),
        CPU_PROVIDER,
    ]

    print("\nLoading ONNX model...")

    load_start = time.perf_counter()

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=providers,
    )

    load_seconds = time.perf_counter() - load_start
    active_providers = session.get_providers()

    print(f"Model loaded in {load_seconds:.3f} seconds.")
    print("Active providers:", active_providers)

    if CUDA_PROVIDER not in active_providers:
        raise RuntimeError(
            "The ONNX model session did not activate "
            "CUDAExecutionProvider."
        )

    return session


def main() -> None:
    """Load the ONNX model and benchmark CUDA inference latency."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing ONNX model: {MODEL_PATH}"
        )

    if not EXTERNAL_DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing ONNX external weights file: "
            f"{EXTERNAL_DATA_PATH}"
        )

    session = create_session()
    print_model_information(session)

    inputs = create_example_inputs()

    print("\nRunning warm-up inference...")

    warmup_outputs = session.run(
        ["logits"],
        inputs,
    )

    warmup_logits = warmup_outputs[0]

    print("Warm-up output shape:", warmup_logits.shape)
    print("Warm-up output dtype:", warmup_logits.dtype)

    inference_times: list[float] = []

    print(
        f"\nRunning {BENCHMARK_RUNS} timed inferences..."
    )

    logits: np.ndarray | None = None

    for run_number in range(1, BENCHMARK_RUNS + 1):
        start_time = time.perf_counter()

        outputs = session.run(
            ["logits"],
            inputs,
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        inference_times.append(elapsed_ms)
        logits = outputs[0]

        print(
            f"Run {run_number}: "
            f"{elapsed_ms:.3f} ms, "
            f"shape={logits.shape}"
        )

    average_ms = float(np.mean(inference_times))
    median_ms = float(np.median(inference_times))
    minimum_ms = float(np.min(inference_times))
    maximum_ms = float(np.max(inference_times))

    print("\nONNX Runtime CUDA test succeeded.")
    print(f"Average latency: {average_ms:.3f} ms")
    print(f"Median latency: {median_ms:.3f} ms")
    print(f"Minimum latency: {minimum_ms:.3f} ms")
    print(f"Maximum latency: {maximum_ms:.3f} ms")

    if logits is not None:
        print("Final logits shape:", logits.shape)
        print("Final logits dtype:", logits.dtype)


if __name__ == "__main__":
    main()