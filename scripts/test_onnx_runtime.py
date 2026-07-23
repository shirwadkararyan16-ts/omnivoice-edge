from pathlib import Path
import time

import numpy as np
import torch
import onnxruntime as ort


MODEL_PATH = Path(
    "models/onnx/omnivoice_forward.onnx"
)


def print_model_information(
    session: ort.InferenceSession,
) -> None:
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


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing ONNX model: {MODEL_PATH}"
        )

    external_data_path = Path(
        f"{MODEL_PATH}.data"
    )

    if not external_data_path.exists():
        raise FileNotFoundError(
            "Missing external weights file: "
            f"{external_data_path}"
        )

    print("ONNX Runtime version:", ort.__version__)
    print(
        "Available providers:",
        ort.get_available_providers(),
    )

    if (
        "CUDAExecutionProvider"
        not in ort.get_available_providers()
    ):
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable."
        )

    session_options = ort.SessionOptions()

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy":
                    "kSameAsRequested",
                "cudnn_conv_algo_search":
                    "DEFAULT",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]

    print("\nLoading ONNX model...")

    load_start = time.perf_counter()

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=providers,
    )

    load_seconds = (
        time.perf_counter() - load_start
    )

    print(
        f"Model loaded in "
        f"{load_seconds:.3f} seconds."
    )

    print(
        "Active providers:",
        session.get_providers(),
    )

    print_model_information(session)

    batch_size = 1
    token_channels = 8
    sequence_length = 8

    input_ids = np.zeros(
        (
            batch_size,
            token_channels,
            sequence_length,
        ),
        dtype=np.int64,
    )

    audio_mask = np.zeros(
        (
            batch_size,
            sequence_length,
        ),
        dtype=np.bool_,
    )

    attention_mask = np.ones(
        (
            batch_size,
            sequence_length,
        ),
        dtype=np.int64,
    )

    position_ids = np.arange(
        sequence_length,
        dtype=np.int64,
    )[None, :]

    inputs = {
        "input_ids": input_ids,
        "audio_mask": audio_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    print("\nRunning warm-up inference...")

    warmup_outputs = session.run(
        ["logits"],
        inputs,
    )

    warmup_logits = warmup_outputs[0]

    print(
        "Warm-up output shape:",
        warmup_logits.shape,
    )

    print(
        "Warm-up output dtype:",
        warmup_logits.dtype,
    )

    inference_times = []

    print("\nRunning five timed inferences...")

    for run_number in range(1, 21):
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

    average_ms = float(
        np.mean(inference_times)
    )

    median_ms = float(
        np.median(inference_times)
    )

    print("\nONNX Runtime CUDA test succeeded.")
    print(f"Average latency: {np.mean(inference_times):.3f} ms")
    print(f"Median latency: {np.median(inference_times):.3f} ms")
    print(f"Minimum latency: {np.min(inference_times):.3f} ms")
    print(f"Maximum latency: {np.max(inference_times):.3f} ms")
    print("Final logits shape:", logits.shape)
    print("Final logits dtype:", logits.dtype)


if __name__ == "__main__":
    main()