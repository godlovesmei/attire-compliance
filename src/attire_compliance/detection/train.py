"""Pelatihan detektor 6 atribut pakaian menggunakan Ultralytics YOLOv12."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from attire_compliance.config import load_all_configs
from attire_compliance.device import resolve_device
from attire_compliance.paths import configure_runtime_env, resolve_project_path


def _validate_data_yaml(data_yaml_path: Path, expected_names: dict[int, str]) -> None:
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"File data.yaml tidak ditemukan: {data_yaml_path}")
    with data_yaml_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    names = raw.get("names")
    if isinstance(names, list):
        parsed = {i: str(v) for i, v in enumerate(names)}
    elif isinstance(names, dict):
        parsed = {int(k): str(v) for k, v in names.items()}
    else:
        raise ValueError(f"Field `names` tidak valid di {data_yaml_path}")
    if parsed != expected_names:
        raise ValueError(
            f"Kelas pada {data_yaml_path} ({parsed}) tidak cocok dengan "
            f"configs/classes.yaml ({expected_names})."
        )


def train_detector(
    *,
    data_yaml: Path | None = None,
    model_name: str | None = None,
    run_name: str | None = None,
    output_weights: Path | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    patience: int | None = None,
    workers: int | None = None,
    device: str | None = None,
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """Latih YOLOv12 sesuai `configs/detector.yaml` dan salin `best.pt` ke `models/detector/`."""
    configure_runtime_env()
    from ultralytics import YOLO

    cfgs = load_all_configs(configs_dir)
    det_cfg = cfgs.detector
    classes_cfg = cfgs.classes

    resolved_data_yaml = (
        data_yaml.resolve() if data_yaml is not None else resolve_project_path(det_cfg.data_yaml)
    )
    _validate_data_yaml(resolved_data_yaml, classes_cfg.id_to_name)

    chosen_model = model_name or det_cfg.model
    pretrained_dir = resolve_project_path(det_cfg.pretrained_dir)
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    pretrained_path = (
        Path(chosen_model)
        if Path(chosen_model).is_file()
        else pretrained_dir / Path(chosen_model).name
    )

    chosen_run = run_name or det_cfg.run_name
    project_dir = resolve_project_path(det_cfg.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    dst_weights = (
        output_weights.resolve()
        if output_weights is not None
        else resolve_project_path(f"models/detector/{chosen_run}.pt")
    )
    dst_weights.parent.mkdir(parents=True, exist_ok=True)

    aug = det_cfg.augmentation
    run_device, use_cuda = resolve_device(device if device is not None else det_cfg.device)
    model = YOLO(str(pretrained_path))
    train_results = model.train(
        data=str(resolved_data_yaml),
        epochs=epochs if epochs is not None else det_cfg.epochs,
        patience=patience if patience is not None else det_cfg.patience,
        batch=batch if batch is not None else det_cfg.batch,
        imgsz=imgsz if imgsz is not None else det_cfg.imgsz,
        workers=workers if workers is not None else det_cfg.workers,
        optimizer=det_cfg.optimizer,
        lr0=det_cfg.lr0,
        lrf=det_cfg.lrf,
        weight_decay=det_cfg.weight_decay,
        warmup_epochs=det_cfg.warmup_epochs,
        amp=bool(det_cfg.amp and use_cuda),
        cache=det_cfg.cache,
        seed=det_cfg.seed,
        deterministic=det_cfg.deterministic,
        device=run_device,
        project=str(project_dir),
        name=chosen_run,
        exist_ok=True,
        hsv_h=aug.hsv_h,
        hsv_s=aug.hsv_s,
        hsv_v=aug.hsv_v,
        degrees=aug.degrees,
        translate=aug.translate,
        scale=aug.scale,
        fliplr=aug.fliplr,
        flipud=aug.flipud,
        mosaic=aug.mosaic,
        mixup=aug.mixup,
    )

    save_dir = Path(getattr(train_results, "save_dir", project_dir / chosen_run))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        best_pt = save_dir / "weights" / "last.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"Checkpoint hasil training tidak ditemukan di {save_dir}")

    shutil.copy2(best_pt, dst_weights)

    metrics_raw = getattr(train_results, "results_dict", {}) or {}
    box_metrics = getattr(train_results, "box", None)
    per_class_ap50: dict[str, float] = {}
    if box_metrics is not None and getattr(box_metrics, "ap50", None) is not None:
        ap50_vals = list(box_metrics.ap50)
        ap_indices = list(getattr(box_metrics, "ap_class_index", range(len(ap50_vals))))
        for idx, val in zip(ap_indices, ap50_vals, strict=False):
            cls_name = classes_cfg.id_to_name.get(int(idx), f"class_{int(idx)}")
            per_class_ap50[cls_name] = round(float(val), 4)

    summary: dict[str, Any] = {
        "run_name": chosen_run,
        "model": str(pretrained_path),
        "data_yaml": str(resolved_data_yaml),
        "save_dir": str(save_dir.resolve()),
        "best_weights": str(dst_weights),
        "epochs": epochs if epochs is not None else det_cfg.epochs,
        "batch": batch if batch is not None else det_cfg.batch,
        "imgsz": imgsz if imgsz is not None else det_cfg.imgsz,
        "metrics": {
            "precision": round(float(metrics_raw.get("metrics/precision(B)", 0.0)), 4),
            "recall": round(float(metrics_raw.get("metrics/recall(B)", 0.0)), 4),
            "mAP50": round(float(metrics_raw.get("metrics/mAP50(B)", 0.0)), 4),
            "mAP50-95": round(float(metrics_raw.get("metrics/mAP50-95(B)", 0.0)), 4),
        },
        "per_class_ap50": per_class_ap50,
    }
    (save_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary
