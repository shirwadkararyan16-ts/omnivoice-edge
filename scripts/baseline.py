import time
from pathlib import Path

import soundfile as sf
import torch
from omnivoice import OmniVoice


OUTPUT_FILE = Path("samples/generated/baseline.wav")


def main() -> None:
    print("Loading model...")

    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",
        dtype=torch.float16,
        load_asr=False,
    )

    print("Model loaded.")
    print("Generating speech...")

    torch.cuda.synchronize()
    start = time.perf_counter()

    audio = model.generate(
        text=(
            "Hello, this audio was generated directly "
            "from my Python script on Windows."
        ),
        language="English",
        instruct="female, young adult, American accent",
        num_step=16,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if not audio:
        raise RuntimeError("OmniVoice returned no audio.")

    samples = audio[0]
    sample_rate = 24000
    audio_duration = len(samples) / sample_rate
    rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sf.write(OUTPUT_FILE, samples, sample_rate)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Generation time: {elapsed:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()