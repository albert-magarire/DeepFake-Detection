"""Inference adapter for the bundled weights and newly trained checkpoints."""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dataset import inference_transform
from utils import load_model as load_checkpoint


def load_model(checkpoint_path=None):
    path = checkpoint_path or os.environ.get("MODEL_CHECKPOINT")
    path = Path(path) if path else Path(__file__).with_name("weights.pth")
    model, checkpoint = load_checkpoint(path, torch.device("cpu"))
    model.image_transform = inference_transform(checkpoint)
    return model


def predict_image(model, image: Image.Image):
    rgb = np.asarray(image.convert("RGB"))
    x = model.image_transform(image=rgb)["image"].unsqueeze(0)
    with torch.inference_mode():
        probability = model(x).sigmoid().item()
    # Preserve the original API: confidence is always the probability of fake.
    return {"label": "Real" if probability < 0.5 else "Fake",
            "confidence": float(probability)}
