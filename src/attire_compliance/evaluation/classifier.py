"""Evaluasi classifier kepatuhan berpakaian EfficientNetV2-S pada test set."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attire_compliance.classification.model import (
    build_classifier_transforms,
    load_classifier_checkpoint,
)
from attire_compliance.classification.train import BinaryAttireDataset
from attire_compliance.config import load_classifier_config
from attire_compliance.paths import configure_runtime_env, resolve_project_path


def compute_detailed_binary_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob_pos: list[float] | None = None,
    *,
    pos_label: int = 1,
) -> dict[str, Any]:
    """Hitung metrik kedua kelas (`sesuai` & `tidak_sesuai`), macro-F1, dan ROC-AUC."""
    if not y_true:
        return {
            "samples": 0,
            "accuracy": 0.0,
            "precision_sesuai": 0.0,
            "recall_sesuai": 0.0,
            "f1_sesuai": 0.0,
            "precision_tidak_sesuai": 0.0,
            "recall_tidak_sesuai": 0.0,
            "f1_tidak_sesuai": 0.0,
            "macro_f1": 0.0,
            "roc_auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    neg_label = 1 - pos_label
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    roc_auc: float | None = None
    if y_prob_pos is not None and len(set(y_true)) == 2:
        roc_auc = round(float(roc_auc_score(y_true, y_prob_pos)), 4)

    return {
        "samples": len(y_true),
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
        "precision_tidak_sesuai": round(
            float(precision_score(y_true, y_pred, pos_label=neg_label, zero_division=0)), 4
        ),
        "recall_tidak_sesuai": round(
            float(recall_score(y_true, y_pred, pos_label=neg_label, zero_division=0)), 4
        ),
        "f1_tidak_sesuai": round(
            float(f1_score(y_true, y_pred, pos_label=neg_label, zero_division=0)), 4
        ),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix_2x2(
    cm: list[list[int]],
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    arr = np.asarray(cm, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=160)
    im = ax.imshow(arr, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title, fontsize=11, fontweight="semibold")

    thresh = arr.max() / 2.0 if arr.size else 0.0
    total = max(1, int(arr.sum()))
    for i in range(2):
        for j in range(2):
            val = int(arr[i, j])
            pct = 100.0 * val / total
            color = "white" if val > thresh else "#1d1d1f"
            ax.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="semibold",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_roc_curve(
    y_true: list[int],
    y_prob_pos: list[float],
    auc_val: float,
    out_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    fig, ax = plt.subplots(figsize=(5.8, 4.6), dpi=160)
    ax.plot(
        fpr,
        tpr,
        color="#0066cc",
        linewidth=2.2,
        label=f"EfficientNetV2-S (AUC = {auc_val:.4f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7a7a7a", linewidth=1.1, label="Random (0.5000)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "Kurva ROC — Classifier EfficientNetV2-S (Test Set)",
        fontsize=11,
        fontweight="semibold",
    )
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_classifier(
    *,
    checkpoint_path: Path | None = None,
    data_dir: Path | None = None,
    split: str = "test",
    output_dir: Path,
    batch_size: int = 32,
    device: str = "cuda",
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluasi EfficientNetV2-S pada split `test`, hasilkan CM 2x2, ROC-AUC, CSV & JSON."""
    configure_runtime_env()
    cfg = load_classifier_config(configs_dir)

    ckpt = (
        checkpoint_path.resolve()
        if checkpoint_path is not None
        else resolve_project_path(cfg.checkpoint_path)
    )
    src_data = data_dir.resolve() if data_dir is not None else resolve_project_path(cfg.data_dir)
    split_dir = src_data / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Direktori split classifier tidak ditemukan: {split_dir}")

    use_cuda = torch.cuda.is_available() and device != "cpu"
    torch_dev = torch.device("cuda:0" if use_cuda else "cpu")

    model, meta = load_classifier_checkpoint(ckpt, map_location=torch_dev)
    model = model.to(torch_dev)
    class_names: list[str] = list(meta["class_names"])
    pos_idx = class_names.index("sesuai")

    eval_tf = build_classifier_transforms(
        int(meta["imgsz"]),
        tuple(meta["mean"]),
        tuple(meta["std"]),
        is_train=False,
        use_letterbox=bool(meta["letterbox"]),
    )
    ds = BinaryAttireDataset(split_dir, class_names, eval_tf)
    if len(ds) == 0:
        raise RuntimeError(f"Dataset classifier kosong pada {split_dir}")

    loader: DataLoader[tuple[torch.Tensor, int]] = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob_pos: list[float] = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(torch_dev, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits = model(images)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_true.extend(targets.tolist())
            y_pred.extend(preds.tolist())
            y_prob_pos.extend(probs[:, pos_idx].tolist())

    metrics = compute_detailed_binary_metrics(
        y_true, y_pred, y_prob_pos, pos_label=pos_idx
    )

    # Pisahkan juga metrik subset lokal Polibatam bila nama file mengandung `local_`
    local_idx = [i for i, (path, _) in enumerate(ds.samples) if path.name.startswith("local_")]
    local_metrics: dict[str, Any] | None = None
    if local_idx:
        local_metrics = compute_detailed_binary_metrics(
            [y_true[i] for i in local_idx],
            [y_pred[i] for i in local_idx],
            [y_prob_pos[i] for i in local_idx],
            pos_label=pos_idx,
        )

    clf_out = output_dir / "classifier"
    clf_out.mkdir(parents=True, exist_ok=True)
    exported_plots: list[str] = []

    cm_path = clf_out / "classifier_confusion_matrix.png"
    plot_confusion_matrix_2x2(
        metrics["confusion_matrix"],
        labels=["Tidak Sesuai", "Sesuai"],
        out_path=cm_path,
        title=f"Confusion Matrix — EfficientNetV2-S ({split} set, n={metrics['samples']})",
    )
    exported_plots.append(cm_path.name)

    if metrics["roc_auc"] is not None:
        roc_path = clf_out / "classifier_roc_curve.png"
        _plot_roc_curve(y_true, y_prob_pos, float(metrics["roc_auc"]), roc_path)
        exported_plots.append(roc_path.name)

    with (clf_out / "classifier_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "is_local", "gt_label", "pred_label", "prob_sesuai", "correct"],
        )
        writer.writeheader()
        for (path, gt), pred, prob in zip(ds.samples, y_pred, y_prob_pos, strict=True):
            writer.writerow(
                {
                    "file": path.name,
                    "is_local": path.name.startswith("local_"),
                    "gt_label": class_names[gt],
                    "pred_label": class_names[pred],
                    "prob_sesuai": round(float(prob), 4),
                    "correct": gt == pred,
                }
            )

    report: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "backbone": meta["backbone"],
        "data_dir": str(src_data),
        "split": split,
        "metrics": metrics,
        "local_subset_metrics": local_metrics,
        "exported_plots": exported_plots,
    }
    (clf_out / "classifier_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
