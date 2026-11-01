"""Evaluate a labeled image folder or classify a single image."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from dataset import ImageDataset, inference_transform
from utils import load_model, run_epoch, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path)
    source.add_argument("--image", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--predictions", type=Path, help="Optional per-image CSV for folder evaluation.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or not 0 <= args.threshold <= 1:
        parser.error("Use batch size >= 1, workers >= 0 and threshold in [0, 1].")
    if args.image and args.predictions:
        parser.error("--predictions is only supported with --data-dir.")
    if args.predictions and args.predictions.exists():
        parser.error("Prediction file already exists; choose a new path.")
    device = select_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    image_size = checkpoint["image_size"]
    transform = inference_transform(checkpoint)
    if args.image:
        with Image.open(args.image) as image:
            rgb = np.asarray(image.convert("RGB"))
        tensor = transform(image=rgb)["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            probability = model(tensor).sigmoid().item()
        print(json.dumps({"fake_probability": probability,
                          "prediction": "fake" if probability >= args.threshold else "real"}, indent=2))
        return
    dataset = ImageDataset(args.data_dir, image_size)
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    metrics, paths, labels, probabilities = run_epoch(model, loader, device, threshold=args.threshold)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "true_label", "predicted_label", "fake_probability"])
            writer.writerows((p, int(y), int(prob >= args.threshold), prob)
                             for p, y, prob in zip(paths, labels, probabilities))


if __name__ == "__main__":
    main()
