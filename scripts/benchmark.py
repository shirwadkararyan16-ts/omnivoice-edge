"""
Display a summary of recorded benchmark results.

This script prints the benchmark results stored in
``benchmarks/results.csv``. It serves as a lightweight utility for
quickly reviewing the recorded ONNX Runtime CUDA and TensorRT FP16
latency measurements without rerunning the full benchmark scripts.

For fresh measurements, use:

    python scripts/test_onnx_runtime.py
    python scripts/test_tensorrt_runtime.py
"""

from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    PROJECT_ROOT
    / "benchmarks"
    / "results.csv"
)


def main() -> None:
    """Print the recorded benchmark results."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {RESULTS_FILE}"
        )

    print("=" * 80)
    print("OmniVoice Edge Benchmark Summary")
    print("=" * 80)

    with RESULTS_FILE.open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            print()
            print(f"Runtime           : {row['Runtime']}")
            print(f"Scope             : {row['Scope']}")
            print(f"Runs              : {row['Runs']}")
            print(f"Average Latency   : {row['Average Latency (ms)']} ms")
            print(f"Median Latency    : {row['Median Latency (ms)']} ms")
            print(f"Minimum Latency   : {row['Minimum Latency (ms)']} ms")
            print(f"Maximum Latency   : {row['Maximum Latency (ms)']} ms")

    print()
    print("=" * 80)
    print("Benchmark summary complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()