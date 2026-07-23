from pathlib import Path
import time

import numpy as np
import torch
import tensorrt as trt
import onnxruntime as ort


MODEL_PATH = Path("models/onnx/omnivoice_forward.onnx")
WEIGHTS_PATH = Path("models/onnx/omnivoice_forward.onnx.data")
CACHE_DIRECTORY = Path("models/tensorrt/cache")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing ONNX model: {MODEL_PATH}"
        )

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing ONNX weights: {WEIGHTS_PATH}"
        )

    CACHE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load CUDA and cuDNN libraries before creating the ORT session.
    ort.preload_dlls()

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("TensorRT:", trt.__version__)
    print("ONNX Runtime:", ort.__version__)
    print(
        "Available providers:",
        ort.get_available_providers(),
    )

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path":
                    str(CACHE_DIRECTORY.resolve()),
                "trt_timing_cache_enable": True,
                "trt_max_workspace_size":
                    512 * 1024 * 1024,
            },
        ),
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy":
                    "kSameAsRequested",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 1

    print()
    print("Creating TensorRT session...")
    print("The first build may take several minutes.")

    start_time = time.perf_counter()

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=providers,
    )

    build_seconds = time.perf_counter() - start_time

    print()
    print(f"Session created in {build_seconds:.3f} seconds.")
    print("Active providers:", session.get_providers())
    print("Provider options:", session.get_provider_options())

    inputs = {
        "input_ids": np.zeros(
            (1, 8, 8),
            dtype=np.int64,
        ),
        "audio_mask": np.zeros(
            (1, 8),
            dtype=np.bool_,
        ),
        "attention_mask": np.ones(
            (1, 8),
            dtype=np.int64,
        ),
        "position_ids": np.arange(
            8,
            dtype=np.int64,
        )[None, :],
    }

    print()
    print("Running warm-up...")

    output = session.run(
        ["logits"],
        inputs,
    )[0]

    print("Warm-up shape:", output.shape)
    print("Warm-up dtype:", output.dtype)

    times = []

    print()
    print("Running five benchmark iterations...")

    for run_number in range(1, 21):
        start_time = time.perf_counter()

        output = session.run(
            ["logits"],
            inputs,
        )[0]

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        times.append(elapsed_ms)

        print(
            f"Run {run_number}: "
            f"{elapsed_ms:.3f} ms"
        )

    print()
    print("Benchmark completed.")
    print(
        f"Average latency: "
        f"{np.mean(times):.3f} ms"
    )
    print(
        f"Median latency: "
        f"{np.median(times):.3f} ms"
    )
    print(
        f"Minimum latency: "
        f"{np.min(times):.3f} ms"
    )
    print(
        f"Maximum latency: "
        f"{np.max(times):.3f} ms"
    )

    print("Output shape:", output.shape)
    print("Cache directory:", CACHE_DIRECTORY.resolve())


if __name__ == "__main__":
    main()