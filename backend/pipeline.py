"""
Reusable speech-generation pipeline utilities for OmniVoice Edge.

This module contains shared data structures and helper functions for
validating generation parameters, measuring inference performance, and
saving generated audio. The FastAPI application can use these utilities
without duplicating pipeline-related logic.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
import uuid

import soundfile as sf
import torch


DEFAULT_SAMPLE_RATE = 24_000
MIN_GENERATION_STEPS = 4
MAX_GENERATION_STEPS = 64


@dataclass(frozen=True)
class GenerationResult:
    """Metadata produced by a completed speech-generation request."""

    output_path: Path
    generation_seconds: float
    audio_seconds: float
    real_time_factor: float


def clamp_generation_steps(steps: int) -> int:
    """
    Constrain the requested generation steps to the supported range.

    Args:
        steps: Requested number of model-generation steps.

    Returns:
        A value between ``MIN_GENERATION_STEPS`` and
        ``MAX_GENERATION_STEPS``.
    """
    return max(
        MIN_GENERATION_STEPS,
        min(steps, MAX_GENERATION_STEPS),
    )


def generate_audio(
    model: Any,
    *,
    text: str,
    language: str,
    instruction: str,
    steps: int,
    output_directory: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> GenerationResult:
    """
    Generate speech, write it to a WAV file, and calculate latency metrics.

    Args:
        model: Loaded OmniVoice-compatible model exposing ``generate``.
        text: Text to synthesize.
        language: Language passed to the model.
        instruction: Voice description or speaking-style instruction.
        steps: Requested number of generation steps.
        output_directory: Directory where the WAV file will be stored.
        sample_rate: Output audio sample rate in hertz.

    Returns:
        Generation metadata including output path, latency, audio duration,
        and real-time factor.

    Raises:
        ValueError: If the text is empty or the model returns no audio.
    """
    normalized_text = text.strip()

    if not normalized_text:
        raise ValueError("Text cannot be empty.")

    validated_steps = clamp_generation_steps(steps)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"generated_{uuid.uuid4().hex}.wav"
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    audio = model.generate(
        text=normalized_text,
        language=language,
        instruct=instruction,
        num_step=validated_steps,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    generation_seconds = time.perf_counter() - start_time

    if not audio:
        raise ValueError("The model returned no audio.")

    samples = audio[0]
    audio_seconds = len(samples) / sample_rate

    real_time_factor = (
        generation_seconds / audio_seconds
        if audio_seconds > 0
        else 0.0
    )

    sf.write(
        output_path,
        samples,
        sample_rate,
    )

    return GenerationResult(
        output_path=output_path,
        generation_seconds=generation_seconds,
        audio_seconds=audio_seconds,
        real_time_factor=real_time_factor,
    )