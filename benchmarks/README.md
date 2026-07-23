# Benchmark Results

This directory contains performance measurements for the OmniVoice Edge inference pipelines.

## Tested Inference Backends

The project evaluates:

- PyTorch baseline inference
- ONNX Runtime with CUDA
- TensorRT through the ONNX Runtime TensorRT Execution Provider

## Observed Performance

| Backend | Average Latency | Median Latency |
|---|---:|---:|
| ONNX Runtime CUDA | ~147 ms | — |
| TensorRT | ~15.2 ms | ~13.3 ms |

The TensorRT execution path provided the lowest latency during testing.

## Important Scope Note

The reported TensorRT acceleration applies only to the exported ONNX forward/logits model wrapper.

It does not represent acceleration of the complete OmniVoice text-to-speech pipeline, including:

- Text preprocessing
- Tokenization
- Autoregressive generation
- Audio decoding
- Audio post-processing

Therefore, the benchmark results should be interpreted as model-component inference measurements rather than complete end-to-end speech-generation latency.

## Benchmark Files

```text
benchmarks/
├── README.md
└── results.csv
```

The `results.csv` file contains the recorded benchmark output.

## Reproducing the Benchmarks

Run the baseline benchmark:

```bash
python scripts/baseline.py
```

Run the general benchmark script:

```bash
python scripts/benchmark.py
```

Run ONNX Runtime CUDA inference:

```bash
python scripts/test_onnx_runtime.py
```

Run TensorRT inference:

```bash
python scripts/test_tensorrt_runtime.py
```

## Results May Vary

Performance depends on:

- GPU model
- CUDA version
- TensorRT version
- ONNX Runtime version
- Input dimensions
- Warm-up iterations
- Benchmark iteration count
- GPU utilization
- System background processes

TensorRT engine and cache files may also need to be regenerated when running on different GPU hardware.