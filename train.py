"""Train the B4 + FFT model on prepared train/ and val/ directories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import ImageDataset, check_path_overlap
from model import DualBranchModel
from utils import build_optimizer, run_epoch, seed_everything, seed_worker, select_device, set_spatial_trainable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=380)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--spatial-lr", type=float, default=5e-6)
    parser.add_argument("--branch-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 2 or args.image_size < 32
            or args.freeze_epochs < 0 or args.workers < 0
            or not 0 <= args.seed < 2**32
            or not 0 < args.spatial_lr < float("inf")
            or not 0 < args.branch_lr < float("inf")
            or not 0 <= args.weight_decay < float("inf")):
        parser.error("Invalid settings: epochs >= 1, batch size >= 2, image size >= 32; use finite positive learning rates.")
    if any((args.output_dir / name).exists() for name in ("best.pt", "last.pt", "config.json")):
        parser.error("Output directory already contains a run; choose a new --output-dir.")

    seed_everything(args.seed)
    device = select_device(args.device)
    train_data = ImageDataset(args.data_dir / "train", args.image_size, True, args.seed)
    val_data = ImageDataset(args.data_dir / "val", args.image_size)
    datasets = [train_data, val_data]
    if (args.data_dir / "test").exists():
        datasets.append(ImageDataset(args.data_dir / "test", args.image_size))
    check_path_overlap(*datasets)
    counts = np.bincount([label for _, label in train_data.samples], minlength=2)
    if np.any(counts == 0) or len(train_data) < args.batch_size:
        parser.error("Training needs both classes and at least one full batch.")
    if len({label for _, label in val_data.samples}) != 2:
        parser.error("Validation needs both real and fake images.")
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        [1.0 / counts[label] for _, label in train_data.samples],
        num_samples=len(train_data), replacement=True, generator=generator)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    # Avoid a singleton final training batch in BatchNorm layers.
    train_loader = DataLoader(train_data, sampler=sampler, drop_last=True,
                              generator=generator, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    model_config = {"backbone": "b4", "spatial_weight": 0.8, "frequency_weight": 0.2}
    model = DualBranchModel(pretrained=not args.no_pretrained, **model_config).to(device)
    set_spatial_trainable(model, args.freeze_epochs == 0)
    optimizer = build_optimizer(model, args.spatial_lr, args.branch_lr, args.weight_decay)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            set_spatial_trainable(model, True)
            optimizer = build_optimizer(model, args.spatial_lr, args.branch_lr, args.weight_decay)
        train_metrics, *_ = run_epoch(model, train_loader, device, optimizer)
        val_metrics, *_ = run_epoch(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_metrics['loss']:.4f} | "
              f"val loss {val_metrics['loss']:.4f} | val macro F1 {val_metrics['macro_f1']:.4f}", flush=True)
        checkpoint = {
            "architecture": "efficientnet_b4_fft", "epoch": epoch,
            "preprocessing": "albumentations",
            "model_config": model_config, "image_size": args.image_size,
            "model_state_dict": model.state_dict(), "train_config": config,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(checkpoint, args.output_dir / "best.pt")
    print(f"Saved checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
