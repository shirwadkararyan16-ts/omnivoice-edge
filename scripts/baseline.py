"""
Run an end-to-end OmniVoice text-to-speech baseline.

This script loads the complete OmniVoice pipeline on CPU, generates a
speech sample, saves the resulting waveform, and reports generation
latency, audio duration, and real-time factor.

The full pipeline is executed on CPU because the available 4 GB GPU does
not have sufficient memory for reliable end-to-end generation. GPU
acceleration is evaluated separately through the exported ONNX forward
subgraph and TensorRT benchmarks.
"""

from pathlib import Path
import time

import soundfile as sf
import torch
from omnivoice import OmniVoice


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    PROJECT_ROOT
    / "samples"
    / "generated"
    / "baseline.wav"
)

MODEL_ID = "k2-fsa/OmniVoice"
DEVICE = "cpu"
MODEL_DTYPE = torch.float32

SAMPLE_RATE = 24_000
GENERATION_STEPS = 16

SAMPLE_TEXT = (
    "Hello, this audio was generated directly "
    "from the OmniVoice Edge baseline script."
)

LANGUAGE = "English"
VOICE_INSTRUCTION = (
    "female, young adult, American accent"
)


def main() -> None:
    """Generate and save an end-to-end OmniVoice speech sample."""
    print("Loading OmniVoice model...")
    print(f"Device: {DEVICE}")
    print(f"Data type: {MODEL_DTYPE}")

    load_start = time.perf_counter()

    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        dtype=MODEL_DTYPE,
        load_asr=False,
    )

    load_seconds = time.perf_counter() - load_start

    print(f"Model loaded in {load_seconds:.2f} seconds.")
    print("Generating speech...")

    generation_start = time.perf_counter()

    audio = model.generate(
        text=SAMPLE_TEXT,
        language=LANGUAGE,
        instruct=VOICE_INSTRUCTION,
        num_step=GENERATION_STEPS,
    )

    generation_seconds = (
        time.perf_counter() - generation_start
    )

    if not audio:
        raise RuntimeError(
            "OmniVoice completed without returning audio."
        )

    samples = audio[0]

    if len(samples) == 0:
        raise RuntimeError(
            "OmniVoice returned an empty audio waveform."
        )

    audio_duration_seconds = (
        len(samples) / SAMPLE_RATE
    )

    real_time_factor = (
        generation_seconds / audio_duration_seconds
        if audio_duration_seconds > 0
        else 0.0
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        OUTPUT_FILE,
        samples,
        SAMPLE_RATE,
    )

    print()
    print("Baseline generation completed.")
    print(f"Saved audio: {OUTPUT_FILE}")
    print(
        f"Generation time: "
        f"{generation_seconds:.2f} seconds"
    )
    print(
        f"Audio duration: "
        f"{audio_duration_seconds:.2f} seconds"
    )
    print(
        f"Real-time factor: "
        f"{real_time_factor:.3f}"
    )


if __name__ == "__main__":
    main()