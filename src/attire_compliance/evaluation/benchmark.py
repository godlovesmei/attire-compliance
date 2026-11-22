"""Benchmark latensi per stage (mean/p95) & FPS end-to-end sesuai standar Bab 4."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attire_compliance.data.merge import IMAGE_EXTS
from attire_compliance.pipeline.pipeline import AttireCompliancePipeline


def _resize_max_side(frame_bgr: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(1, max(h, w)))
    if scale >= 1.0:
        return frame_bgr
    return cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))))


def _stage_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "std_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }


def _plot_latency_breakdown(stages: dict[str, dict[str, float]], out_path: Path) -> None:
    keys = ["person_ms", "attribute_ms", "classifier_ms", "fusion_ms", "total_ms"]
    labels = [
        "Stage 0\nPerson",
        "Stage 1\nAtribut",
        "Stage 2\nEffNetV2",
        "Stage 3\nFusion",
        "Total\nEnd-to-End",
    ]
    means = [stages[k]["mean_ms"] for k in keys]
    p95s = [stages[k]["p95_ms"] for k in keys]

    x = np.arange(len(keys))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=160)
    b1 = ax.bar(x - width / 2, means, width, label="Mean (ms)", color="#0066cc")
    b2 = ax.bar(x + width / 2, p95s, width, label="P95 (ms)", color="#272729")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.6,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="semibold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latensi (ms)")
    ax.set_title(
        "Breakdown Latensi Pipeline Hybrid per Stage (Mean vs P95)",
        fontsize=11.5,
        fontweight="semibold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_latency_benchmark(
    pipeline: AttireCompliancePipeline,
    *,
    sample_dir: Path,
    output_dir: Path,
    warmup_frames: int = 20,
    measure_frames: int = 300,
    max_side: int = 640,
) -> dict[str, Any]:
    """Jalankan benchmark latensi per stage & FPS (warmup >= 20 frame, ukur >= 300 frame)."""
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Direktori frame benchmark tidak ditemukan: {sample_dir}")

    # Prioritaskan gambar lokal Polibatam bila ada, lalu sisanya
    all_imgs = sorted(
        p for p in sample_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    local_first = [p for p in all_imgs if p.name.startswith(("local_", "almet_", "nonalmet_"))]
    selected_paths = (local_first if local_first else all_imgs)[:30]
    frames: list[np.ndarray] = []
    for p in selected_paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(_resize_max_side(img, max_side=max_side))

    if not frames:
        raise RuntimeError(f"Tidak ada gambar valid untuk benchmark di {sample_dir}")

    pipeline.reset_tracking()
    for i in range(warmup_frames):
        pipeline.process_frame(frames[i % len(frames)], track=False, use_smoothing=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    stage_series: dict[str, list[float]] = {
        "person_ms": [],
        "attribute_ms": [],
        "classifier_ms": [],
        "fusion_ms": [],
        "total_ms": [],
    }
    person_counts: list[int] = []

    for i in range(measure_frames):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res = pipeline.process_frame(frames[i % len(frames)], track=False, use_smoothing=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stage_series["person_ms"].append(res.timing.person_ms)
        stage_series["attribute_ms"].append(res.timing.attribute_ms)
        stage_series["classifier_ms"].append(res.timing.classifier_ms)
        stage_series["fusion_ms"].append(res.timing.fusion_ms)
        stage_series["total_ms"].append(res.timing.total_ms)
        person_counts.append(len(res.persons))

    stages = {k: _stage_stats(v) for k, v in stage_series.items()}
    mean_total = stages["total_ms"]["mean_ms"]
    p95_total = stages["total_ms"]["p95_ms"]
    fps_mean = round(1000.0 / mean_total, 2) if mean_total > 0 else 0.0
    fps_p95 = round(1000.0 / p95_total, 2) if p95_total > 0 else 0.0

    hardware = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if torch.cuda.is_available()
            else 0.0
        ),
        "fp16_enabled": pipeline.half,
    }

    bench_out = output_dir / "benchmark"
    bench_out.mkdir(parents=True, exist_ok=True)
    plot_path = bench_out / "benchmark_stage_latency.png"
    _plot_latency_breakdown(stages, plot_path)

    with (bench_out / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "mean_ms", "std_ms", "p50_ms", "p95_ms"])
        writer.writeheader()
        for stage_name, stats in stages.items():
            writer.writerow({"stage": stage_name, **stats})

    report: dict[str, Any] = {
        "warmup_frames": warmup_frames,
        "measure_frames": measure_frames,
        "unique_sample_frames": len(frames),
        "max_side_px": max_side,
        "avg_persons_per_frame": round(float(np.mean(person_counts)), 2),
        "fps_mean": fps_mean,
        "fps_p95": fps_p95,
        "stages": stages,
        "hardware": hardware,
        "exported_plots": [plot_path.name],
    }
    (bench_out / "benchmark_latency.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
