"""
System and GPU telemetry utilities for OmniVoice Edge.

This module collects process memory usage, total system memory usage,
CPU utilization, and NVIDIA GPU statistics for the web dashboard.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional, TypedDict

import psutil


class GpuTelemetry(TypedDict):
    """GPU metrics returned by the NVIDIA System Management Interface."""

    gpu_utilization_percent: Optional[float]
    gpu_memory_used_mb: Optional[float]
    gpu_memory_total_mb: Optional[float]
    gpu_temperature_c: Optional[float]


class TelemetryData(GpuTelemetry):
    """Combined process, system, CPU, and GPU telemetry measurements."""

    process_memory_mb: float
    system_memory_percent: float
    cpu_percent: float


class SystemTelemetry:
    """Collect process, system, CPU, and NVIDIA GPU measurements."""

    def __init__(self) -> None:
        """Create a telemetry collector for the current Python process."""
        self.process = psutil.Process(os.getpid())

    @staticmethod
    def _empty_gpu_metrics() -> GpuTelemetry:
        """
        Return unavailable GPU metric values.

        Returns:
            A dictionary containing ``None`` for every GPU measurement.
        """
        return {
            "gpu_utilization_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
            "gpu_temperature_c": None,
        }

    @staticmethod
    def _read_nvidia_smi() -> GpuTelemetry:
        """
        Read GPU utilization, memory, and temperature from ``nvidia-smi``.

        Returns:
            GPU metrics reported by the first NVIDIA GPU. If ``nvidia-smi``
            is unavailable or returns invalid output, all GPU values are
            returned as ``None``.
        """
        command = [
            "nvidia-smi",
            "--query-gpu="
            "utilization.gpu,"
            "memory.used,"
            "memory.total,"
            "temperature.gpu",
            "--format=csv,noheader,nounits",
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )

            output_lines = result.stdout.strip().splitlines()

            if not output_lines:
                return SystemTelemetry._empty_gpu_metrics()

            values = [
                value.strip()
                for value in output_lines[0].split(",")
            ]

            if len(values) != 4:
                return SystemTelemetry._empty_gpu_metrics()

            return {
                "gpu_utilization_percent": float(values[0]),
                "gpu_memory_used_mb": float(values[1]),
                "gpu_memory_total_mb": float(values[2]),
                "gpu_temperature_c": float(values[3]),
            }

        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ValueError,
        ):
            return SystemTelemetry._empty_gpu_metrics()

    def collect(self) -> TelemetryData:
        """
        Collect current process, system, CPU, and GPU measurements.

        Returns:
            A dictionary containing memory, CPU, GPU utilization,
            GPU memory, and GPU temperature values.
        """
        process_memory_bytes = self.process.memory_info().rss

        telemetry: TelemetryData = {
            "process_memory_mb": process_memory_bytes / (1024 * 1024),
            "system_memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=None),
            **self._read_nvidia_smi(),
        }

        return telemetry