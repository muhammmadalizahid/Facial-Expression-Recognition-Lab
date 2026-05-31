import base64
import io
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from model_loader import DEFAULT_CLASSES, DEFAULT_IMG_SIZE, ModelManager

app = FastAPI(title="Facial Expression Recognition API", version="1.0.0")
model_manager = ModelManager()
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadModelRequest(BaseModel):
    model_path: str


class PredictBase64Request(BaseModel):
    image: str


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model_manager.is_loaded,
        "device": model_manager.device,
        "model_path": model_manager.model_path,
        "classes": model_manager.classes if model_manager.is_loaded else DEFAULT_CLASSES,
        "img_size": model_manager.img_size if model_manager.is_loaded else DEFAULT_IMG_SIZE,
        "metadata": model_manager.metadata,
    }


@app.post("/load-model")
def load_model(payload: LoadModelRequest) -> dict[str, Any]:
    requested = Path(payload.model_path)
    candidate_paths: list[Path] = []

    if requested.is_absolute():
        candidate_paths.append(requested)
    else:
        candidate_paths.extend(
            [
                Path.cwd() / requested,
                BACKEND_DIR / requested,
                PROJECT_ROOT / requested,
                BACKEND_DIR / "models" / requested.name,
            ]
        )

    model_path = next((p.resolve() for p in candidate_paths if p.exists()), None)
    if model_path is None:
        searched = ", ".join(str(p) for p in candidate_paths)
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {payload.model_path}. Searched: {searched}",
        )

    try:
        info = model_manager.load_model(str(model_path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {exc}") from exc

    return {
        "message": "Model loaded successfully",
        "model_loaded": True,
        "model_path": info.model_path,
        "device": info.device,
        "classes": info.classes,
        "img_size": info.img_size,
        "architecture": info.architecture,
        "metadata": info.metadata,
    }


def _predict_from_image_bytes(raw_bytes: bytes) -> dict[str, Any]:
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Call /load-model first.")

    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    try:
        return model_manager.predict_pil(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image payload")
    return _predict_from_image_bytes(raw)


@app.post("/predict-base64")
def predict_base64(payload: PredictBase64Request) -> dict[str, Any]:
    image_data = payload.image.strip()
    if not image_data:
        raise HTTPException(status_code=400, detail="Image data is empty")

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc

    return _predict_from_image_bytes(raw)
