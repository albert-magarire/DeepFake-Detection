"""Seeding, optimization, evaluation and checkpoint helpers."""

import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.utils.data.get_worker_info().dataset.transform.set_random_seed(seed)


def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def set_spatial_trainable(model, trainable):
    for parameter in model.spatial.parameters():
        parameter.requires_grad_(trainable)


def build_optimizer(model, spatial_lr=5e-6, branch_lr=1e-4, weight_decay=1e-4):
    groups = [
        {"params": model.freq_branch.parameters(), "lr": branch_lr},
        {"params": model.classifier.parameters(), "lr": branch_lr},
    ]
    spatial = [p for p in model.spatial.parameters() if p.requires_grad]
    if spatial:
        groups.insert(0, {"params": spatial, "lr": spatial_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def compute_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if (labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels)
            or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()
            or not 0 <= threshold <= 1):
        raise ValueError("Expected binary labels, finite probabilities and a threshold in [0, 1].")
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0)
    return {
        "count": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "macro_f1": float(f1.mean()),
        "threshold": threshold,
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(("real", "fake"))
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def run_epoch(model, loader, device, optimizer=None, threshold=0.5):
    training = optimizer is not None
    model.train(training)
    labels_all, probabilities_all, paths_all = [], [], []
    loss_sum = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.set_grad_enabled(training):
        for images, labels, paths in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; check inputs and learning rates.")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            labels_all.extend(labels.detach().cpu().tolist())
            probabilities_all.extend(logits.detach().sigmoid().cpu().tolist())
            paths_all.extend(paths)
    metrics = compute_metrics(labels_all, probabilities_all, threshold)
    metrics["loss"] = loss_sum / len(labels_all)
    return metrics, paths_all, labels_all, probabilities_all


def load_model(checkpoint_path, device):
    from model import DualBranchModel

    # The original checkpoint contains NumPy scalar/array training metadata.
    # Allow only these known NumPy types instead of enabling arbitrary pickle loading.
    allowed = [np._core.multiarray.scalar, np._core.multiarray._reconstruct,
               np.ndarray, np.dtype, np.dtypes.Float64DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a checkpoint containing model_state_dict.")
    architecture = checkpoint.get("architecture")
    if architecture not in (None, "efficientnet_b4_fft", "efficientnet_b0_fft"):
        raise ValueError("Unsupported checkpoint architecture.")
    width = checkpoint["model_state_dict"]["classifier.0.weight"].shape[1]
    backbone = {1536: "b0", 2048: "b4"}.get(width)
    if backbone is None or (architecture and architecture != f"efficientnet_{backbone}_fft"):
        raise ValueError("Checkpoint dimensions do not match the declared architecture.")
    config = dict(checkpoint.get("model_config", {}))
    if config.get("backbone", backbone) != backbone:
        raise ValueError("Checkpoint backbone configuration does not match its weights.")
    config["backbone"] = backbone
    model = DualBranchModel(**config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    checkpoint = dict(checkpoint)
    checkpoint["architecture"] = f"efficientnet_{backbone}_fft"
    checkpoint.setdefault("image_size", 380)
    checkpoint.setdefault("preprocessing", "torchvision_resize" if architecture is None else "albumentations")
    if checkpoint["preprocessing"] not in ("torchvision_resize", "albumentations"):
        raise ValueError("Unknown checkpoint preprocessing.")
    return model, checkpoint
