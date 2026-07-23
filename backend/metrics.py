"""
Performance metric utilities for OmniVoice Edge.

This module provides reusable helpers for calculating real-time factor
and formatting speech-generation measurements returned by the API.
"""


def calculate_real_time_factor(
    generation_seconds: float,
    audio_seconds: float,
) -> float:
    """
    Calculate the real-time factor for generated audio.

    Real-time factor is defined as generation time divided by generated
    audio duration. Values below 1.0 indicate faster-than-real-time
    generation.

    Args:
        generation_seconds: Time spent generating the audio.
        audio_seconds: Duration of the generated audio.

    Returns:
        The calculated real-time factor.

    Raises:
        ValueError: If either duration is negative.
    """
    if generation_seconds < 0:
        raise ValueError("Generation time cannot be negative.")

    if audio_seconds < 0:
        raise ValueError("Audio duration cannot be negative.")

    if audio_seconds == 0:
        return 0.0

    return generation_seconds / audio_seconds


def format_generation_metrics(
    generation_seconds: float,
    audio_seconds: float,
) -> dict[str, float]:
    """
    Create a consistent dictionary of generation performance metrics.

    Args:
        generation_seconds: Time spent generating the audio.
        audio_seconds: Duration of the generated audio.

    Returns:
        A dictionary containing generation time, audio duration, and RTF.
    """
    return {
        "generation_seconds": generation_seconds,
        "audio_seconds": audio_seconds,
        "real_time_factor": calculate_real_time_factor(
            generation_seconds,
            audio_seconds,
        ),
    }