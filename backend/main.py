"""
FastAPI application for the OmniVoice Edge text-to-speech playground.

This module:

- Loads the OmniVoice model during application startup.
- Serves the browser-based frontend.
- Exposes health and telemetry endpoints.
- Generates speech audio from user-provided text.
- Returns generation latency, audio duration, and real-time factor
  measurements through HTTP response headers.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import time
import uuid

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from omnivoice import OmniVoice
from pydantic import BaseModel

from backend.telemetry import SystemTelemetry


MODEL_ID = "k2-fsa/OmniVoice"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIRECTORY = PROJECT_ROOT / "frontend"
OUTPUT_DIRECTORY = PROJECT_ROOT / "samples" / "generated"
SAMPLE_RATE = 24_000

model: OmniVoice | None = None
telemetry = SystemTelemetry()


class GenerateRequest(BaseModel):
    """Request body accepted by the speech-generation endpoint."""

    text: str
    language: str = "English"
    instruction: str = "female, young adult, American accent"
    steps: int = 16


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Load the OmniVoice model at startup and release GPU memory at shutdown.

    The model is initialized once when the FastAPI application starts so
    subsequent generation requests can reuse the same loaded model.

    Args:
        _app: FastAPI application instance supplied by the framework.

    Yields:
        Control back to FastAPI while the application is running.
    """
    global model

    print("Loading OmniVoice model...")

    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )

    print("OmniVoice model loaded.")

    yield

    model = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="OmniVoice Edge API",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIRECTORY),
    name="static",
)


@app.get("/")
def home() -> FileResponse:
    """
    Serve the main frontend page.

    Returns:
        The OmniVoice Edge playground HTML file.
    """
    return FileResponse(FRONTEND_DIRECTORY / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    """
    Return the current API, model, and CUDA availability status.

    Returns:
        A dictionary containing service health and GPU information.
    """
    cuda_available = torch.cuda.is_available()

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "cuda_available": cuda_available,
        "gpu": (
            torch.cuda.get_device_name(0)
            if cuda_available
            else None
        ),
    }


@app.get("/telemetry")
def get_telemetry() -> dict:
    """
    Collect the latest system and GPU telemetry measurements.

    Returns:
        Current telemetry information reported by ``SystemTelemetry``.
    """
    return telemetry.collect()


@app.post("/generate")
def generate_speech(request: GenerateRequest) -> FileResponse:
    """
    Generate a WAV audio file from the submitted text.

    The requested diffusion step count is constrained to the supported
    range of 4 through 64. Generation duration and real-time factor are
    measured and returned through custom HTTP response headers.

    Args:
        request: Text, language, voice instruction, and step settings.

    Returns:
        A WAV audio file containing the generated speech.

    Raises:
        HTTPException: If the model is unavailable, the input text is
            empty, or the model does not return audio.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    steps = max(4, min(request.steps, 64))

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIRECTORY
        / f"generated_{uuid.uuid4().hex}.wav"
    )

    # Synchronize CUDA before timing to exclude unfinished GPU operations.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    audio = model.generate(
        text=text,
        language=request.language,
        instruct=request.instruction,
        num_step=steps,
    )

    # Wait for GPU generation to finish before calculating elapsed time.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    generation_seconds = time.perf_counter() - start_time

    if not audio:
        raise HTTPException(
            status_code=500,
            detail="OmniVoice returned no audio.",
        )

    samples = audio[0]
    audio_seconds = len(samples) / SAMPLE_RATE

    rtf = (
        generation_seconds / audio_seconds
        if audio_seconds > 0
        else 0.0
    )

    sf.write(
        output_file,
        samples,
        SAMPLE_RATE,
    )

    response = FileResponse(
        path=output_file,
        media_type="audio/wav",
        filename="generated.wav",
    )

    response.headers["X-Generation-Seconds"] = (
        f"{generation_seconds:.3f}"
    )
    response.headers["X-Audio-Seconds"] = (
        f"{audio_seconds:.3f}"
    )
    response.headers["X-RTF"] = f"{rtf:.3f}"

    return response