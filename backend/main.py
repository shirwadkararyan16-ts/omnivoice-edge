from contextlib import asynccontextmanager
from backend.telemetry import SystemTelemetry
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


MODEL_ID = "k2-fsa/OmniVoice"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIRECTORY = PROJECT_ROOT / "frontend"
OUTPUT_DIRECTORY = PROJECT_ROOT / "samples" / "generated"
SAMPLE_RATE = 24000

model: OmniVoice | None = None
telemetry = SystemTelemetry()


class GenerateRequest(BaseModel):
    text: str
    language: str = "English"
    instruction: str = "female, young adult, American accent"
    steps: int = 16


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("Loading OmniVoice model...")

    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.float16,
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
def home():
    return FileResponse(FRONTEND_DIRECTORY / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }

@app.get("/telemetry")
def get_telemetry() -> dict:
    return telemetry.collect()

@app.post("/generate")
def generate_speech(request: GenerateRequest):
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

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    audio = model.generate(
        text=text,
        language=request.language,
        instruct=request.instruction,
        num_step=steps,
    )

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