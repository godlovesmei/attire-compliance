"""Evaluasi detektor YOLOv12 pada test set (keseluruhan & subset kampus lokal Polibatam)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attire_compliance.config import load_all_configs
from attire_compliance.paths import configure_runtime_env, resolve_project_path


def _plot_per_class_ap50(per_class: list[dict[str, Any]], out_path: Path, title: str) -> None:
    names = [row["class_name"] for row in per_class]
    vals = [float(row["ap50"]) for row in per_class]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=160)
    bars = ax.bar(names, vals, color="#0066cc", edgecolor="#1d1d1f", linewidth=0.6)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("AP@50")
    ax.set_title(title, fontsize=12, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=18)
    for bar, val in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(1.03, val + 0.02),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _extract_val_metrics(
    val_res: Any,
    id_to_name: dict[int, str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    raw = getattr(val_res, "results_dict", {}) or {}
    overall = {
        "precision": round(float(raw.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(raw.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(raw.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50-95": round(float(raw.get("metrics/mAP50-95(B)", 0.0)), 4),
    }

    box = getattr(val_res, "box", None)
    per_class: list[dict[str, Any]] = []
    if box is not None and getattr(box, "ap50", None) is not None:
        ap50_list = list(box.ap50)
        ap_list = list(getattr(box, "ap", [0.0] * len(ap50_list)))
        p_list = list(getattr(box, "p", [0.0] * len(ap50_list)))
        r_list = list(getattr(box, "r", [0.0] * len(ap50_list)))
        f1_list = list(getattr(box, "f1", [0.0] * len(ap50_list)))
        indices = list(getattr(box, "ap_class_index", range(len(ap50_list))))
        for idx, ap50, ap95, p, r, f1 in zip(
            indices, ap50_list, ap_list, p_list, r_list, f1_list, strict=False
        ):
            cid = int(idx)
            per_class.append(
                {
                    "class_id": cid,
                    "class_name": id_to_name.get(cid, f"class_{cid}"),
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "f1": round(float(f1), 4),
                    "ap50": round(float(ap50), 4),
                    "ap50_95": round(float(ap95), 4),
                }
            )
    return overall, per_class


def _build_local_subset_yaml(
    data_yaml: Path, split: str, work_dir: Path
) -> tuple[Path | None, int]:
    """Buat YAML sementara yang hanya menunjuk gambar `local_*` pada split target."""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    ds_root = Path(cfg.get("path", data_yaml.parent))
    split_rel = cfg.get(split, f"images/{split}")
    img_dir = ds_root / split_rel
    if not img_dir.is_dir():
        return None, 0

    local_imgs = sorted(p.resolve() for p in img_dir.iterdir() if p.name.startswith("local_"))
    if not local_imgs:
        return None, 0

    work_dir.mkdir(parents=True, exist_ok=True)
    txt_path = work_dir / f"local_{split}_images.txt"
    txt_path.write_text("\n".join(str(p) for p in local_imgs) + "\n", encoding="utf-8")

    subset_yaml = work_dir / f"local_{split}_subset.yaml"
    payload = {
        "path": str(ds_root.resolve()),
        "train": cfg.get("train", "images/train"),
        "val": str(txt_path.resolve()),
        "test": str(txt_path.resolve()),
        "names": cfg["names"],
    }
    subset_yaml.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return subset_yaml, len(local_imgs)


def evaluate_detector(
    *,
    weights_path: Path | None = None,
    data_yaml: Path | None = None,
    split: str = "test",
    output_dir: Path,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    include_local_subset: bool = True,
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluasi YOLOv12 pada split `test`, ekspor kurva Ultralytics, CSV, dan JSON."""
    configure_runtime_env()
    from ultralytics import YOLO

    cfgs = load_all_configs(configs_dir)
    id_to_name = cfgs.classes.id_to_name

    resolved_weights = (
        weights_path.resolve()
        if weights_path is not None
        else resolve_project_path(cfgs.app.models.attribute_detector)
    )
    if not resolved_weights.is_file():
        raise FileNotFoundError(f"Bobot detektor tidak ditemukan: {resolved_weights}")

    resolved_yaml = (
        data_yaml.resolve()
        if data_yaml is not None
        else resolve_project_path(cfgs.detector.data_yaml)
    )
    if not resolved_yaml.is_file():
        raise FileNotFoundError(f"File data.yaml tidak ditemukan: {resolved_yaml}")

    det_out = output_dir / "detector"
    det_out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(resolved_weights))
    val_res = model.val(
        data=str(resolved_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=True,
        project=str(det_out),
        name="overall_plots",
        exist_ok=True,
        verbose=False,
    )
    overall, per_class = _extract_val_metrics(val_res, id_to_name)

    # Salin plot utama Ultralytics (CM, PR, F1) ke root folder detector agar mudah ditemukan
    plots_dir = Path(getattr(val_res, "save_dir", det_out / "overall_plots"))
    exported_plots: list[str] = []
    for fname in (
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "PR_curve.png",
        "F1_curve.png",
    ):
        src = plots_dir / fname
        if src.is_file():
            dst = det_out / f"detector_{fname}"
            shutil.copy2(src, dst)
            exported_plots.append(dst.name)

    if per_class:
        _plot_per_class_ap50(
            per_class,
            det_out / "detector_ap50_per_class.png",
            f"YOLOv12 AP@50 per Kelas ({split} set)",
        )
        exported_plots.append("detector_ap50_per_class.png")

        with (det_out / "detector_per_class.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "class_id",
                    "class_name",
                    "precision",
                    "recall",
                    "f1",
                    "ap50",
                    "ap50_95",
                ],
            )
            writer.writeheader()
            writer.writerows(per_class)

    # Evaluasi tambahan khusus domain lokal Polibatam pada split test
    local_summary: dict[str, Any] | None = None
    if include_local_subset:
        subset_yaml, local_count = _build_local_subset_yaml(
            resolved_yaml, split, det_out / "_local_subset_tmp"
        )
        if subset_yaml is not None and local_count > 0:
            local_res = model.val(
                data=str(subset_yaml),
                split="test",
                imgsz=imgsz,
                batch=min(batch, local_count),
                device=device,
                plots=True,
                project=str(det_out),
                name="local_subset_plots",
                exist_ok=True,
                verbose=False,
            )
            loc_overall, loc_per_class = _extract_val_metrics(local_res, id_to_name)
            local_summary = {
                "images": local_count,
                "overall": loc_overall,
                "per_class": loc_per_class,
            }
            if loc_per_class:
                _plot_per_class_ap50(
                    loc_per_class,
                    det_out / "detector_local_ap50_per_class.png",
                    f"YOLOv12 AP@50 per Kelas — Subset Lokal Polibatam ({local_count} gambar)",
                )
                exported_plots.append("detector_local_ap50_per_class.png")

    report: dict[str, Any] = {
        "weights": str(resolved_weights),
        "data_yaml": str(resolved_yaml),
        "split": split,
        "imgsz": imgsz,
        "overall": overall,
        "per_class": per_class,
        "local_subset": local_summary,
        "exported_plots": exported_plots,
    }
    (det_out / "detector_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
