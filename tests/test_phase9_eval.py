"""Unit test offline (CPU/stub) untuk Fase 9: modul evaluasi & laporan Bab 4."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from attire_compliance.config import load_all_configs
from attire_compliance.detection.predictor import Detection
from attire_compliance.evaluation.benchmark import run_latency_benchmark
from attire_compliance.evaluation.classifier import (
    compute_detailed_binary_metrics,
    plot_confusion_matrix_2x2,
)
from attire_compliance.evaluation.detector import _build_local_subset_yaml
from attire_compliance.evaluation.end_to_end import evaluate_end_to_end
from attire_compliance.evaluation.runner import _render_bab4_markdown
from attire_compliance.pipeline.fusion import FusionDecision
from attire_compliance.pipeline.pipeline import FrameResult, FrameTiming, PersonAnalysis


class _EvalStubPipeline:
    """Stub pipeline deterministik untuk menguji end-to-end ablation & benchmark tanpa GPU."""

    def __init__(self) -> None:
        self.configs = load_all_configs()
        self.app_cfg = self.configs.app
        self.classes_cfg = self.configs.classes
        self.half = False

    def reset_tracking(self) -> None:
        return None

    def process_frame(self, frame_bgr: np.ndarray, **_kwargs: Any) -> FrameResult:
        h, w = frame_bgr.shape[:2]
        # Prediksi berdasarkan intensitas piksel: frame terang -> sesuai, gelap -> kaos oblong
        bright = float(frame_bgr.mean()) > 120.0
        if bright:
            attrs = [
                Detection(1, "jas_almamater", 0.92, (20.0, 20.0, w - 20.0, h * 0.55)),
                Detection(3, "bawahan_panjang", 0.90, (20.0, h * 0.50, w - 20.0, h * 0.88)),
                Detection(4, "sepatu", 0.88, (25.0, h * 0.85, w - 25.0, h - 5.0)),
            ]
            prob = 0.93
        else:
            attrs = [
                Detection(2, "kaos_oblong", 0.91, (20.0, 20.0, w - 20.0, h * 0.50)),
                Detection(3, "bawahan_panjang", 0.85, (20.0, h * 0.50, w - 20.0, h * 0.88)),
                Detection(4, "sepatu", 0.82, (25.0, h * 0.85, w - 25.0, h - 5.0)),
            ]
            prob = 0.12

        decision = FusionDecision(
            status="SESUAI" if bright else "TIDAK SESUAI",
            reason="stub",
            mode="fusion",
            rule_status="SESUAI" if bright else "TIDAK SESUAI",
            prob_sesuai=prob,
            prob_tidak_sesuai=1.0 - prob,
            active_violations=() if bright else ("kaos_oblong",),
            missing_required=(),
        )
        person = PersonAnalysis(
            track_id=1,
            bbox=(10.0, 10.0, float(w - 10), float(h - 4)),
            confidence=0.95,
            full_body_visible=True,
            attributes=attrs,
            raw_decision=decision,
            smoothed_status=decision.status,
        )
        return FrameResult(
            frame_width=w,
            frame_height=h,
            persons=[person],
            unassigned_attributes=[],
            timing=FrameTiming(4.0, 6.0, 2.0, 0.1, 12.1, 82.6),
        )


def test_detailed_binary_metrics_and_cm_plot(tmp_path: Path) -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_prob = [0.1, 0.6, 0.85, 0.95]
    m = compute_detailed_binary_metrics(y_true, y_pred, y_prob, pos_label=1)
    assert m["samples"] == 4
    assert m["accuracy"] == 0.75
    assert m["recall_sesuai"] == 1.0
    assert m["recall_tidak_sesuai"] == 0.5
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]

    cm_png = tmp_path / "cm.png"
    plot_confusion_matrix_2x2(m["confusion_matrix"], ["Tidak Sesuai", "Sesuai"], cm_png, "CM Test")
    assert cm_png.is_file() and cm_png.stat().st_size > 1000


def test_local_subset_yaml_builder(tmp_path: Path) -> None:
    test_dir = tmp_path / "images" / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "local_almet_01.jpg").write_bytes(b"jpg")
    (test_dir / "fashionpedia_01.jpg").write_bytes(b"jpg")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"path": str(tmp_path), "test": "images/test", "names": {0: "a"}}),
        encoding="utf-8",
    )
    subset_yaml, count = _build_local_subset_yaml(data_yaml, "test", tmp_path / "work")
    assert subset_yaml is not None and subset_yaml.is_file()
    assert count == 1


def test_end_to_end_ablation_and_benchmark_and_markdown(tmp_path: Path) -> None:
    img_dir = tmp_path / "det" / "images" / "test"
    lbl_dir = tmp_path / "det" / "labels" / "test"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # 1 frame lokal terang (sesuai) + 1 frame publik gelap (kaos oblong -> tidak sesuai)
    bright_img = np.full((300, 160, 3), 200, dtype=np.uint8)
    dark_img = np.full((300, 160, 3), 40, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "local_almet_01.jpg"), bright_img)
    cv2.imwrite(str(img_dir / "fp_01.jpg"), dark_img)

    (lbl_dir / "local_almet_01.txt").write_text(
        "1 0.5 0.30 0.6 0.40\n3 0.5 0.68 0.6 0.36\n4 0.5 0.92 0.5 0.12\n",
        encoding="utf-8",
    )
    (lbl_dir / "fp_01.txt").write_text(
        "2 0.5 0.30 0.6 0.40\n3 0.5 0.68 0.6 0.36\n4 0.5 0.92 0.5 0.12\n",
        encoding="utf-8",
    )

    stub = _EvalStubPipeline()
    out_dir = tmp_path / "report"

    e2e = evaluate_end_to_end(
        stub,  # type: ignore[arg-type]
        detection_dir=tmp_path / "det",
        split="test",
        output_dir=out_dir,
        project_root=tmp_path,
    )
    assert e2e["evaluated_images"] == 2
    assert e2e["evaluated_persons"] == 2
    assert e2e["local_persons"] == 1
    assert e2e["ablation_overall"]["fusion"]["accuracy"] == 1.0
    assert (out_dir / "end_to_end" / "ablation_table.csv").is_file()
    assert (out_dir / "end_to_end" / "end_to_end_ablation.png").is_file()

    bench = run_latency_benchmark(
        stub,  # type: ignore[arg-type]
        sample_dir=img_dir,
        output_dir=out_dir,
        warmup_frames=2,
        measure_frames=10,
    )
    assert bench["measure_frames"] == 10
    assert bench["fps_mean"] > 0
    assert (out_dir / "benchmark" / "benchmark_stage_latency.png").is_file()

    md = _render_bab4_markdown(
        {
            "generated_at": "2026-11-22T21:40:00",
            "output_dir": str(out_dir),
            "detector": None,
            "classifier": None,
            "end_to_end": e2e,
            "benchmark": bench,
        }
    )
    assert "## 3. Evaluasi End-to-End & Studi Ablation" in md
    assert "## 4. Benchmark Kecepatan & Latensi per Stage" in md


def test_ui_design_system_strict_compliance() -> None:
    """Verifikasi UI mematuhi aturan sistem desain (single accent, flat cards, 100% offline)."""
    root = Path(__file__).resolve().parents[1] / "src" / "attire_compliance" / "app"
    html = (root / "templates" / "index.html").read_text(encoding="utf-8")
    css = (root / "static" / "styles.css").read_text(encoding="utf-8")
    js = (root / "static" / "app.js").read_text(encoding="utf-8")

    # 100% offline: tidak ada referensi CDN eksternal
    for text in (html, css, js):
        assert "http://" not in text and "https://" not in text

    # Token warna inti DESIGN.md hadir
    for token in ("#0066cc", "#0071e3", "#2997ff", "#1d1d1f", "#f5f5f7", "#272729", "#000000"):
        assert token in css

    # Larangan mutlak DESIGN.md: zero gradient, zero weight 500, hanya 1 drop-shadow
    assert "gradient(" not in css
    assert re.search(r"font-weight\s*:\s*500\b", css) is None
    assert len(re.findall(r"box-shadow\s*:", css)) == 1

    # Aturan tipografi & mikro-interaksi DESIGN.md
    assert "font-size: 17px;" in css
    assert "line-height: 1.47;" in css
    assert "transform: scale(0.95);" in css
    assert "backdrop-filter:" in css

    # Zero redundansi kontrol UI/UX (1 tombol Mulai Deteksi, 1 mode switcher, 1 Capture)
    assert html.count('id="btn-start"') == 1
    assert "btn-start-hero" not in html
    assert html.count('data-mode="webcam"') == 1
    assert html.count('id="btn-capture"') == 1
