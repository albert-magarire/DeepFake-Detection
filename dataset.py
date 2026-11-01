"""RGB image loading and augmentation for preassigned dataset splits."""

from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ("real", "fake")


def make_transform(image_size=380, training=False, seed=42):
    if training:
        steps = [
            A.RandomResizedCrop(size=(image_size, image_size),
                                scale=(0.6, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 3), p=0.5),
                A.MotionBlur(blur_limit=(3, 3), p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=0.3),
            A.ImageCompression(quality_range=(40, 95), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
                          hue=0.02, p=0.5),
        ]
    else:
        steps = [A.Resize(image_size, image_size)]
    return A.Compose(steps + [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], seed=seed)


class ImageDataset(Dataset):
    """Read root/real/** and root/fake/**; labels are real=0, fake=1."""

    def __init__(self, root, image_size=380, training=False, seed=42):
        self.root = Path(root).resolve()
        self.transform = make_transform(image_size, training, seed)
        self.samples = []
        for label, name in enumerate(CLASS_NAMES):
            folder = self.root / name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class folder: {folder}")
            paths = sorted(p for p in folder.rglob("*") if p.is_file()
                           and p.suffix.lower() in IMAGE_EXTENSIONS
                           and not any(s.startswith('.') for s in p.relative_to(folder).parts))
            self.samples.extend((str(p.resolve()), label) for p in paths)
        if not self.samples:
            raise ValueError(f"No supported images found in {self.root}")
        if len({p for p, _ in self.samples}) != len(self.samples):
            raise ValueError("The dataset contains repeated resolved image paths.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        tensor = self.transform(image=rgb)["image"]
        return tensor, torch.tensor(label, dtype=torch.float32), path


def check_path_overlap(*datasets):
    """Reject repeated paths; source-video and identity grouping is done upstream."""
    seen = set()
    for dataset in datasets:
        paths = {p for p, _ in dataset.samples}
        overlap = seen & paths
        if overlap:
            raise ValueError(f"Image path appears in multiple splits: {min(overlap)}")
        seen.update(paths)
