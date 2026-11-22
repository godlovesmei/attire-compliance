"""Pelatihan classifier kepatuhan berpakaian EfficientNetV2-S (PyTorch + timm + AMP)."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from attire_compliance.classification.model import (
    build_classifier_transforms,
    create_classifier_model,
    save_classifier_checkpoint,
)
from attire_compliance.config import load_classifier_config
from attire_compliance.paths import configure_runtime_env, resolve_project_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class BinaryAttireDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset folder `{split}/{tidak_sesuai,sesuai}/*.jpg` dengan indeks kelas eksplisit."""

    def __init__(self, split_dir: Path, class_names: list[str], transform: Any) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.class_names = list(class_names)
        self.transform = transform

        for idx, cls_name in enumerate(self.class_names):
            cls_dir = split_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((img_path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        return self.transform(rgb), target


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_binary_metrics(
    y_true: list[int],
    y_pred: list[int],
    *,
    pos_label: int = 1,
) -> dict[str, Any]:
    """Hitung Accuracy, Precision, Recall, F1 (pos & macro), dan Confusion Matrix 2x2."""
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision_sesuai": 0.0,
            "recall_sesuai": 0.0,
            "f1_sesuai": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_sesuai": round(
            float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)), 4
        ),
        "recall_sesuai": round(
            float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)), 4
        ),
        "f1_sesuai": round(
            float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)), 4
        ),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": cm,
    }


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    pos_label: int,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)
            batch_size = int(targets.size(0))
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(1, total_count)
    return avg_loss, compute_binary_metrics(y_true, y_pred, pos_label=pos_label)


def train_classifier(
    *,
    data_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    run_name: str = "effnetv2s_compliance_v1",
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    patience: int | None = None,
    pretrained: bool | None = None,
    device_override: str | None = None,
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """Latih EfficientNetV2-S dengan AMP, warmup + cosine LR, dan early stopping val macro-F1."""
    configure_runtime_env()
    cfg = load_classifier_config(configs_dir)
    _seed_everything(cfg.seed)

    src_data = data_dir.resolve() if data_dir is not None else resolve_project_path(cfg.data_dir)
    out_ckpt = (
        checkpoint_path.resolve()
        if checkpoint_path is not None
        else resolve_project_path(cfg.checkpoint_path)
    )
    run_dir = resolve_project_path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dev_str = device_override or cfg.device
    use_cuda = torch.cuda.is_available() and dev_str != "cpu"
    device = torch.device("cuda:0" if use_cuda else "cpu")
    use_amp = bool(cfg.amp and use_cuda)

    model, norm_meta = create_classifier_model(
        backbone=cfg.backbone,
        num_classes=len(cfg.class_names),
        pretrained=cfg.pretrained if pretrained is None else pretrained,
    )
    model = model.to(device)

    train_tf = build_classifier_transforms(
        cfg.imgsz, norm_meta["mean"], norm_meta["std"], is_train=True, use_letterbox=cfg.letterbox
    )
    eval_tf = build_classifier_transforms(
        cfg.imgsz, norm_meta["mean"], norm_meta["std"], is_train=False, use_letterbox=cfg.letterbox
    )

    train_ds = BinaryAttireDataset(src_data / "train", cfg.class_names, train_tf)
    val_ds = BinaryAttireDataset(src_data / "val", cfg.class_names, eval_tf)
    test_ds = BinaryAttireDataset(src_data / "test", cfg.class_names, eval_tf)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Dataset classifier kosong di {src_data} "
            f"(train={len(train_ds)}, val={len(val_ds)}). Jalankan build-classification dulu."
        )

    bs = batch_size if batch_size is not None else cfg.batch_size
    workers = cfg.num_workers if use_cuda else 0
    train_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        pin_memory=use_cuda,
    )
    val_loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_cuda,
    )
    test_loader: DataLoader[tuple[torch.Tensor, int]] | None = (
        DataLoader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=workers,
            pin_memory=use_cuda,
        )
        if len(test_ds) > 0
        else None
    )

    # Inverse-frequency class weight untuk menangani ketidakseimbangan kelas
    counts = np.bincount([t for _, t in train_ds.samples], minlength=len(cfg.class_names))
    safe_counts = np.maximum(counts, 1)
    weights_arr = safe_counts.sum() / (len(cfg.class_names) * safe_counts)
    class_weights = torch.tensor(weights_arr, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()

    total_epochs = epochs if epochs is not None else cfg.epochs
    max_patience = patience if patience is not None else cfg.patience
    base_lr = lr if lr is not None else cfg.lr
    pos_label = cfg.class_names.index("sesuai")

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=cfg.weight_decay)
    warmup = min(cfg.warmup_epochs, max(0, total_epochs - 1))

    def _lr_lambda(epoch_idx: int) -> float:
        if warmup > 0 and epoch_idx < warmup:
            return float(epoch_idx + 1) / float(warmup)
        progress = (epoch_idx - warmup) / max(1, total_epochs - warmup)
        return float(0.01 + 0.99 * 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_epoch = 0
    best_val_metrics: dict[str, Any] = {}
    no_improve = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs_cur = int(targets.size(0))
            running_loss += float(loss.item()) * bs_cur
            seen += bs_cur

        scheduler.step()
        train_loss = running_loss / max(1, seen)
        val_loss, val_metrics = _evaluate_loader(
            model, val_loader, eval_criterion, device, use_amp, pos_label
        )
        val_f1 = float(val_metrics["macro_f1"])

        history.append(
            {
                "epoch": epoch,
                "lr": round(float(optimizer.param_groups[0]["lr"]), 6),
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_f1,
            }
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_val_metrics = val_metrics
            no_improve = 0
            save_classifier_checkpoint(
                out_ckpt,
                model=model,
                backbone=cfg.backbone,
                class_names=cfg.class_names,
                imgsz=cfg.imgsz,
                letterbox=cfg.letterbox,
                mean=norm_meta["mean"],
                std=norm_meta["std"],
                metrics={"best_epoch": best_epoch, "val": val_metrics},
            )
        else:
            no_improve += 1
            if max_patience > 0 and no_improve >= max_patience:
                break

    # Evaluasi checkpoint terbaik pada test set jika tersedia
    test_metrics: dict[str, Any] | None = None
    if test_loader is not None and out_ckpt.is_file():
        payload = torch.load(out_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        _, test_metrics = _evaluate_loader(
            model, test_loader, eval_criterion, device, use_amp, pos_label
        )

    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary: dict[str, Any] = {
        "run_name": run_name,
        "backbone": cfg.backbone,
        "checkpoint_path": str(out_ckpt),
        "data_dir": str(src_data),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "train_class_counts": {
            cfg.class_names[i]: int(counts[i]) for i in range(len(cfg.class_names))
        },
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
