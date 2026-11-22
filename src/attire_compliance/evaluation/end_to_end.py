"""Evaluasi End-to-End Hybrid Pipeline & Tabel Ablation (`rule_only`/`classifier_only`/`fusion`)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attire_compliance.config import FusionMode
from attire_compliance.data.classification import (
    _load_fashionpedia_raw_categories,
    _yolo_to_detections,
    prelabel_crop_from_gt,
)
from attire_compliance.data.merge import IMAGE_EXTS, parse_yolo_label_file
from attire_compliance.evaluation.classifier import (
    compute_detailed_binary_metrics,
    plot_confusion_matrix_2x2,
)
from attire_compliance.paths import resolve_project_path
from attire_compliance.pipeline.association import PersonBox, associate_attributes_to_persons
from attire_compliance.pipeline.fusion import fuse_decision
from attire_compliance.pipeline.pipeline import AttireCompliancePipeline
from attire_compliance.pipeline.rules import evaluate_rules

ABLATION_MODES: tuple[FusionMode, ...] = ("rule_only", "classifier_only", "fusion")


def _plot_ablation_comparison(
    ablation_metrics: dict[str, dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    modes = list(ABLATION_MODES)
    metric_keys = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("recall_tidak_sesuai", "Recall Tidak Sesuai"),
        ("f1_sesuai", "F1 Sesuai"),
    ]
    x = np.arange(len(metric_keys))
    width = 0.25
    colors = ["#7a7a7a", "#272729", "#0066cc"]

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    for idx, mode in enumerate(modes):
        vals = [float(ablation_metrics[mode].get(k, 0.0)) for k, _ in metric_keys]
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, vals, width=width, label=mode, color=colors[idx])
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(1.04, val + 0.015),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="semibold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_keys])
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Skor")
    ax.set_title(title, fontsize=11.5, fontweight="semibold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_end_to_end(
    pipeline: AttireCompliancePipeline,
    *,
    detection_dir: Path | None = None,
    split: str = "test",
    output_dir: Path,
    max_images: int | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluasi pipeline Stage 0–3 pada test set & bandingkan 3 mode ablation secara fair."""
    det_root = (
        detection_dir.resolve()
        if detection_dir is not None
        else resolve_project_path(pipeline.configs.detector.data_yaml, project_root).parent
    )
    img_dir = det_root / "images" / split
    lbl_dir = det_root / "labels" / split
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"Direktori test deteksi tidak ditemukan: {img_dir}")

    fp_meta = _load_fashionpedia_raw_categories(
        resolve_project_path(pipeline.configs.sources.fashionpedia.output_dir, project_root)
    )
    id_to_name = pipeline.classes_cfg.id_to_name
    num_classes = len(pipeline.classes_cfg.names)

    records_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in ABLATION_MODES}
    image_paths = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]

    evaluated_images = 0
    for img_path in image_paths:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        gt_labels, _ = parse_yolo_label_file(lbl_path, num_classes=num_classes)
        if not gt_labels:
            continue
        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            continue
        evaluated_images += 1
        img_h, img_w = frame_bgr.shape[:2]

        # Jalankan pipeline nyata sekali per gambar (evaluasi statis tanpa smoothing/incomplete)
        frame_res = pipeline.process_frame(
            frame_bgr,
            track=False,
            use_smoothing=False,
            allow_incomplete=False,
        )
        if not frame_res.persons:
            continue

        # Asosiasikan GT atribut ke person yang terdeteksi untuk membangun GT status person
        gt_dets = _yolo_to_detections(gt_labels, img_w, img_h, id_to_name)
        person_boxes = [
            PersonBox(track_id=p.track_id, bbox=p.bbox, confidence=p.confidence)
            for p in frame_res.persons
        ]
        gt_per_person, _ = associate_attributes_to_persons(
            person_boxes, gt_dets, min_ioa=pipeline.app_cfg.thresholds.association_min_ioa
        )

        has_shorts = False
        for key_part in ("fp_train_", "fp_val_"):
            pos = img_path.stem.find(key_part)
            if pos != -1:
                has_shorts = "shorts" in fp_meta.get(img_path.stem[pos:], set())
                break

        is_local = img_path.name.startswith("local_")

        for person in frame_res.persons:
            gt_attrs = {d.class_name for d in gt_per_person.get(person.track_id, [])}
            if not gt_attrs:
                continue
            gt_label_str, _ = prelabel_crop_from_gt(
                gt_attrs,
                almamater_overrides_tshirt=pipeline.app_cfg.fusion.almamater_overrides_tshirt,
                has_shorts_meta=has_shorts,
            )
            if gt_label_str is None:
                continue

            gt_int = 1 if gt_label_str == "sesuai" else 0
            pred_attrs = [a.class_name for a in person.attributes]
            rule_eval = evaluate_rules(
                pred_attrs,
                full_body_visible=True,
                almamater_overrides_tshirt=pipeline.app_cfg.fusion.almamater_overrides_tshirt,
                allow_incomplete=False,
            )
            prob_sesuai = person.raw_decision.prob_sesuai

            for mode in ABLATION_MODES:
                decision = fuse_decision(
                    rule_eval,
                    prob_sesuai=prob_sesuai,
                    mode=mode,
                    classifier_veto_threshold=pipeline.app_cfg.fusion.classifier_veto_threshold,
                    full_body_visible=True,
                    allow_incomplete=False,
                )
                pred_int = 1 if decision.status == "SESUAI" else 0
                records_by_mode[mode].append(
                    {
                        "image": img_path.name,
                        "is_local": is_local,
                        "track_id": person.track_id,
                        "gt_int": gt_int,
                        "pred_int": pred_int,
                        "prob_sesuai": prob_sesuai,
                        "gt_attrs": ";".join(sorted(gt_attrs)),
                        "pred_attrs": ";".join(sorted(set(pred_attrs))),
                        "reason": decision.reason,
                    }
                )

    ablation_overall: dict[str, dict[str, Any]] = {}
    ablation_local: dict[str, dict[str, Any]] = {}

    for mode in ABLATION_MODES:
        rows = records_by_mode[mode]
        probs = [r["prob_sesuai"] for r in rows]
        has_probs = all(p is not None for p in probs) and len(probs) > 0
        ablation_overall[mode] = compute_detailed_binary_metrics(
            [r["gt_int"] for r in rows],
            [r["pred_int"] for r in rows],
            [float(p) for p in probs] if has_probs else None,
            pos_label=1,
        )

        loc_rows = [r for r in rows if r["is_local"]]
        loc_probs = [r["prob_sesuai"] for r in loc_rows]
        loc_has_probs = all(p is not None for p in loc_probs) and len(loc_probs) > 0
        ablation_local[mode] = compute_detailed_binary_metrics(
            [r["gt_int"] for r in loc_rows],
            [r["pred_int"] for r in loc_rows],
            [float(p) for p in loc_probs] if loc_has_probs else None,
            pos_label=1,
        )

    e2e_out = output_dir / "end_to_end"
    e2e_out.mkdir(parents=True, exist_ok=True)
    exported_plots: list[str] = []

    # Ekspor tabel ablation CSV
    with (e2e_out / "ablation_table.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scope",
            "mode",
            "samples",
            "accuracy",
            "precision_sesuai",
            "recall_sesuai",
            "f1_sesuai",
            "recall_tidak_sesuai",
            "f1_tidak_sesuai",
            "macro_f1",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        scopes = (("overall", ablation_overall), ("local_polibatam", ablation_local))
        for scope_name, mapping in scopes:
            for mode in ABLATION_MODES:
                m = mapping[mode]
                writer.writerow(
                    {
                        "scope": scope_name,
                        "mode": mode,
                        "samples": m["samples"],
                        "accuracy": m["accuracy"],
                        "precision_sesuai": m["precision_sesuai"],
                        "recall_sesuai": m["recall_sesuai"],
                        "f1_sesuai": m["f1_sesuai"],
                        "recall_tidak_sesuai": m["recall_tidak_sesuai"],
                        "f1_tidak_sesuai": m["f1_tidak_sesuai"],
                        "macro_f1": m["macro_f1"],
                    }
                )

    # Simpan contoh kasus gagal (mode fusion) untuk analisis error Bab 4
    failures = [r for r in records_by_mode["fusion"] if r["gt_int"] != r["pred_int"]]
    with (e2e_out / "failure_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "is_local",
                "track_id",
                "gt_label",
                "pred_label",
                "prob_sesuai",
                "gt_attrs",
                "pred_attrs",
                "reason",
            ],
        )
        writer.writeheader()
        for r in failures:
            writer.writerow(
                {
                    "image": r["image"],
                    "is_local": r["is_local"],
                    "track_id": r["track_id"],
                    "gt_label": "SESUAI" if r["gt_int"] == 1 else "TIDAK SESUAI",
                    "pred_label": "SESUAI" if r["pred_int"] == 1 else "TIDAK SESUAI",
                    "prob_sesuai": r["prob_sesuai"] if r["prob_sesuai"] is not None else "",
                    "gt_attrs": r["gt_attrs"],
                    "pred_attrs": r["pred_attrs"],
                    "reason": r["reason"],
                }
            )

    if ablation_overall["fusion"]["samples"] > 0:
        ablation_plot = e2e_out / "end_to_end_ablation.png"
        n_eval = ablation_overall["fusion"]["samples"]
        _plot_ablation_comparison(
            ablation_overall,
            ablation_plot,
            f"Perbandingan Mode Ablation End-to-End ({split} set, n={n_eval} orang)",
        )
        exported_plots.append(ablation_plot.name)

        cm_plot = e2e_out / "end_to_end_cm_fusion.png"
        plot_confusion_matrix_2x2(
            ablation_overall["fusion"]["confusion_matrix"],
            labels=["Tidak Sesuai", "Sesuai"],
            out_path=cm_plot,
            title=f"Confusion Matrix End-to-End Fusion (n={ablation_overall['fusion']['samples']})",
        )
        exported_plots.append(cm_plot.name)

    report: dict[str, Any] = {
        "split": split,
        "evaluated_images": evaluated_images,
        "evaluated_persons": ablation_overall["fusion"]["samples"],
        "local_persons": ablation_local["fusion"]["samples"],
        "failure_count_fusion": len(failures),
        "ablation_overall": ablation_overall,
        "ablation_local": ablation_local,
        "exported_plots": exported_plots,
    }
    (e2e_out / "end_to_end_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
