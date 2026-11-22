"""Orkestrator evaluasi menyeluruh Bab 4 (`attire evaluate all`) + generator `BAB4_SUMMARY.md`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from attire_compliance.evaluation.benchmark import run_latency_benchmark
from attire_compliance.evaluation.classifier import evaluate_classifier
from attire_compliance.evaluation.detector import evaluate_detector
from attire_compliance.evaluation.end_to_end import ABLATION_MODES, evaluate_end_to_end
from attire_compliance.paths import resolve_project_path
from attire_compliance.pipeline.pipeline import AttireCompliancePipeline


def _render_bab4_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Ringkasan Hasil Evaluasi Otomatis — Bahan Bab 4",
        "",
        f"- **Tanggal:** `{summary['generated_at']}`",
        f"- **Folder Laporan:** `{summary['output_dir']}`",
        "",
    ]

    det = summary.get("detector")
    if det:
        ov = det["overall"]
        lines.extend(
            [
                "## 1. Evaluasi Detektor Atribut (YOLOv12 — Test Set)",
                "",
                f"- **Bobot:** `{det['weights']}`",
                (
                    f"- **Overall:** Precision `{ov['precision']:.4f}` · "
                    f"Recall `{ov['recall']:.4f}` · **mAP@50 `{ov['mAP50']:.4f}`** · "
                    f"mAP@50-95 `{ov['mAP50-95']:.4f}`"
                ),
                "",
                "| Kelas | Precision | Recall | F1 | AP@50 | AP@50-95 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in det.get("per_class", []):
            lines.append(
                f"| `{row['class_name']}` | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['ap50']:.4f} | {row['ap50_95']:.4f} |"
            )
        loc = det.get("local_subset")
        if loc:
            lov = loc["overall"]
            lines.extend(
                [
                    "",
                    f"**Subset Kampus Lokal Polibatam ({loc['images']} gambar test):** "
                    f"P `{lov['precision']:.4f}` · R `{lov['recall']:.4f}` · "
                    f"**mAP@50 `{lov['mAP50']:.4f}`** · mAP@50-95 `{lov['mAP50-95']:.4f}`",
                ]
            )
        lines.append("")

    clf = summary.get("classifier")
    if clf:
        cm = clf["metrics"]
        lines.extend(
            [
                "## 2. Evaluasi Classifier Kepatuhan (EfficientNetV2-S — Test Set)",
                "",
                f"- **Checkpoint:** `{clf['checkpoint']}` (`{clf['backbone']}`)",
                f"- **Sampel Test:** {cm['samples']} crop full-body",
                "",
                "| Metrik | Nilai |",
                "|---|---:|",
                f"| Accuracy | {cm['accuracy']:.4f} |",
                f"| Precision (Sesuai) | {cm['precision_sesuai']:.4f} |",
                f"| Recall (Sesuai) | {cm['recall_sesuai']:.4f} |",
                f"| F1 (Sesuai) | {cm['f1_sesuai']:.4f} |",
                f"| Recall (Tidak Sesuai) | {cm['recall_tidak_sesuai']:.4f} |",
                f"| Macro-F1 | {cm['macro_f1']:.4f} |",
                f"| ROC-AUC | {cm['roc_auc'] if cm['roc_auc'] is not None else 'N/A'} |",
                f"| Confusion Matrix ([[TN, FP], [FN, TP]]) | `{cm['confusion_matrix']}` |",
                "",
            ]
        )

    e2e = summary.get("end_to_end")
    if e2e:
        lines.extend(
            [
                "## 3. Evaluasi End-to-End & Studi Ablation",
                "",
                (
                    f"- **Gambar dievaluasi:** {e2e['evaluated_images']} · "
                    f"**Total person GT:** {e2e['evaluated_persons']} "
                    f"(lokal Polibatam: {e2e['local_persons']})"
                ),
                "",
                "| Scope | Mode | n | Accuracy | F1 (Sesuai) | Recall (Tidak Sesuai) | Macro-F1 |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        scopes = (("Overall", "ablation_overall"), ("Lokal Polibatam", "ablation_local"))
        for scope_label, key in scopes:
            for mode in ABLATION_MODES:
                m = e2e[key][mode]
                lines.append(
                    f"| {scope_label} | `{mode}` | {m['samples']} | {m['accuracy']:.4f} | "
                    f"{m['f1_sesuai']:.4f} | {m['recall_tidak_sesuai']:.4f} | {m['macro_f1']:.4f} |"
                )
        lines.append("")

    bench = summary.get("benchmark")
    if bench:
        hw = bench["hardware"]
        lines.extend(
            [
                "## 4. Benchmark Kecepatan & Latensi per Stage",
                "",
                (
                    f"- **Hardware:** `{hw['gpu_name']}` ({hw['vram_gib']} GiB VRAM, "
                    f"CUDA `{hw['cuda_version']}`, FP16=`{hw['fp16_enabled']}`)"
                ),
                (
                    f"- **Konfigurasi:** warmup {bench['warmup_frames']} frame, "
                    f"ukur {bench['measure_frames']} frame (`max_side={bench['max_side_px']}px`, "
                    f"rata-rata `{bench['avg_persons_per_frame']}` orang/frame)"
                ),
                (
                    f"- **Kecepatan End-to-End:** **{bench['fps_mean']} FPS** (mean) · "
                    f"**{bench['fps_p95']} FPS** (p95)"
                ),
                "",
                "| Stage | Mean (ms) | Std (ms) | P50 (ms) | P95 (ms) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for st_name, st in bench["stages"].items():
            lines.append(
                f"| `{st_name}` | {st['mean_ms']:.2f} | {st['std_ms']:.2f} | "
                f"{st['p50_ms']:.2f} | {st['p95_ms']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def run_full_evaluation(
    *,
    tag: str = "v1_bab4",
    detector_weights: Path | None = None,
    classifier_weights: Path | None = None,
    data_yaml: Path | None = None,
    classifier_data_dir: Path | None = None,
    split: str = "test",
    warmup_frames: int = 20,
    measure_frames: int = 300,
    output_dir: Path | None = None,
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """Jalankan evaluasi menyeluruh (detektor, classifier, end-to-end ablation, benchmark)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (
        output_dir.resolve()
        if output_dir is not None
        else resolve_project_path(f"outputs/reports/{stamp}_{tag}")
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    det_report = evaluate_detector(
        weights_path=detector_weights,
        data_yaml=data_yaml,
        split=split,
        output_dir=report_dir,
        configs_dir=configs_dir,
    )
    clf_report = evaluate_classifier(
        checkpoint_path=classifier_weights,
        data_dir=classifier_data_dir,
        split=split,
        output_dir=report_dir,
        configs_dir=configs_dir,
    )

    pipeline = AttireCompliancePipeline(
        configs_dir=configs_dir,
        attribute_weights=Path(det_report["weights"]),
        classifier_weights=Path(clf_report["checkpoint"]),
    )
    pipeline.warmup()

    det_root = Path(det_report["data_yaml"]).parent
    e2e_report = evaluate_end_to_end(
        pipeline,
        detection_dir=det_root,
        split=split,
        output_dir=report_dir,
    )
    bench_report = run_latency_benchmark(
        pipeline,
        sample_dir=det_root / "images" / split,
        output_dir=report_dir,
        warmup_frames=warmup_frames,
        measure_frames=measure_frames,
    )

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(report_dir),
        "detector": det_report,
        "classifier": clf_report,
        "end_to_end": e2e_report,
        "benchmark": bench_report,
    }
    (report_dir / "full_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (report_dir / "BAB4_SUMMARY.md").write_text(
        _render_bab4_markdown(summary), encoding="utf-8"
    )
    return summary
