from __future__ import annotations

import os
import subprocess

import psutil


class SystemTelemetry:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())

    @staticmethod
    def _read_nvidia_smi() -> dict:
        try:
            command = [
                "nvidia-smi",
                "--query-gpu="
                "utilization.gpu,"
                "memory.used,"
                "memory.total,"
                "temperature.gpu",
                "--format=csv,noheader,nounits",
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )

            first_line = result.stdout.strip().splitlines()[0]

            values = [
                value.strip()
                for value in first_line.split(",")
            ]

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
            IndexError,
            ValueError,
        ):
            return {
                "gpu_utilization_percent": None,
                "gpu_memory_used_mb": None,
                "gpu_memory_total_mb": None,
                "gpu_temperature_c": None,
            }

    def collect(self) -> dict:
        process_memory = self.process.memory_info().rss

        data = {
            "process_memory_mb": process_memory / 1024 / 1024,
            "system_memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=None),
        }

        data.update(self._read_nvidia_smi())

        return data