"""FastAPI image upload endpoint."""

from contextlib import asynccontextmanager
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.model import load_model, predict_image


@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model()
    yield


app = FastAPI(lifespan=lifespan)
origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip().rstrip("/") for origin in origins.split(",") if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Welcome to the Deepfake Detection API. Use the /predict endpoint to analyze images."}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    image_bytes = file.file.read(10 * 1024 * 1024 + 1)
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail="Upload a valid image.") from error
    return predict_image(app.state.model, rgb)
