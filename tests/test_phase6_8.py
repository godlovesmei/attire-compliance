"""Unit test untuk Fase 6 (classifier utils), Fase 7 (smoothing), & Fase 8 (FastAPI + WS)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from attire_compliance.app.main import create_app
from attire_compliance.classification.model import (
    crop_person_with_padding,
    letterbox_rgb,
    load_classifier_checkpoint,
    save_classifier_checkpoint,
)
from attire_compliance.config import load_all_configs
from attire_compliance.data.classification import prelabel_crop_from_gt
from attire_compliance.detection.predictor import Detection
from attire_compliance.device import autocast_context, resolve_device, resolve_torch_device
from attire_compliance.pipeline.fusion import FusionDecision
from attire_compliance.pipeline.pipeline import FrameResult, FrameTiming, PersonAnalysis
from attire_compliance.pipeline.smoothing import TemporalSmoother


def test_letterbox_and_crop_preserve_aspect_ratio() -> None:
    portrait = Image.new("RGB", (100, 250), (200, 100, 50))
    boxed = letterbox_rgb(portrait, size=224)
    assert boxed.size == (224, 224)
    # Tepi kiri/kanan harus warna padding (114, 114, 114), tengah warna gambar asli
    assert boxed.getpixel((0, 112)) == (114, 114, 114)
    assert boxed.getpixel((112, 112)) == (200, 100, 50)

    frame_bgr = np.full((400, 300, 3), 255, dtype=np.uint8)
    crop = crop_person_with_padding(frame_bgr, (100, 50, 200, 350), pad_ratio=0.10)
    assert crop.size == (120, 360)


def test_prelabel_crop_from_gt_rules() -> None:
    lbl, _ = prelabel_crop_from_gt({"jas_almamater", "kaos_oblong", "bawahan_panjang", "sepatu"})
    assert lbl == "sesuai"

    lbl_bad, _ = prelabel_crop_from_gt({"kaos_oblong", "bawahan_panjang", "sepatu"})
    assert lbl_bad == "tidak_sesuai"

    lbl_shorts, _ = prelabel_crop_from_gt({"pakaian_berkerah", "sepatu"}, has_shorts_meta=True)
    assert lbl_shorts == "tidak_sesuai"

    lbl_ambig, _ = prelabel_crop_from_gt({"sepatu"})
    assert lbl_ambig is None


def test_classifier_checkpoint_roundtrip(tmp_path: Path) -> None:
    from torch import nn

    # Uji roundtrip dengan backbone kecil tanpa download bobot internet
    ckpt = tmp_path / "clf_test.pt"
    tiny_backbone = "resnet18"
    import timm

    model = timm.create_model(tiny_backbone, pretrained=False, num_classes=2)
    assert isinstance(model, nn.Module)
    save_classifier_checkpoint(
        ckpt,
        model=model,
        backbone=tiny_backbone,
        class_names=["tidak_sesuai", "sesuai"],
        imgsz=224,
        letterbox=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        metrics={"val": {"macro_f1": 0.91}},
    )
    loaded, meta = load_classifier_checkpoint(ckpt)
    assert meta["class_names"] == ["tidak_sesuai", "sesuai"]
    assert meta["imgsz"] == 224
    assert isinstance(loaded, nn.Module)


def test_temporal_smoother_suppresses_single_frame_flicker() -> None:
    smoother = TemporalSmoother(window_size=5, max_stale_frames=10)
    for _ in range(4):
        smoother.step_frame()
        assert smoother.update(1, "SESUAI") == "SESUAI"
    # 1 frame glitch tidak langsung mengubah status mayoritas
    smoother.step_frame()
    assert smoother.update(1, "TIDAK SESUAI") == "SESUAI"
    smoother.reset()
    smoother.step_frame()
    assert smoother.update(1, "TIDAK SESUAI") == "TIDAK SESUAI"


def test_device_resolution_cpu_fallback_and_cuda(monkeypatch: Any) -> None:
    import torch

    # Simulasikan laptop klien tanpa CUDA GPU: request "cuda" atau "0" HARUS fallback ke "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda") == ("cpu", False)
    assert resolve_device("0") == ("cpu", False)
    assert resolve_device("cuda:0") == ("cpu", False)
    assert resolve_device("auto") == ("cpu", False)
    assert resolve_device("cpu") == ("cpu", False)
    t_dev, use_cuda = resolve_torch_device("cuda")
    assert t_dev == torch.device("cpu")
    assert use_cuda is False
    with autocast_context(True):
        pass

    # Simulasikan workstation dengan CUDA GPU tersedia
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == (0, True)
    assert resolve_device("0") == (0, True)
    assert resolve_device("cuda:1") == (1, True)
    assert resolve_device("cpu") == ("cpu", False)
    t_dev_gpu, use_cuda_gpu = resolve_torch_device("cuda")
    assert t_dev_gpu == torch.device("cuda:0")
    assert use_cuda_gpu is True


class _StubPipeline:
    """Stub ringan untuk menguji seluruh endpoint FastAPI & WebSocket secara offline."""

    def __init__(self) -> None:
        self.configs = load_all_configs()
        self.app_cfg = self.configs.app
        self.classes_cfg = self.configs.classes
        self.person_weights_path = Path("yolo12n.pt")
        self.attribute_weights_path = Path("yolo12n_attire_v0.pt")
        self.classifier_weights_path = None
        self.classifier = None
        self.reset_called = 0

    def reset_tracking(self) -> None:
        self.reset_called += 1

    def process_frame(self, frame_bgr: np.ndarray, **_kwargs: Any) -> FrameResult:
        h, w = frame_bgr.shape[:2]
        decision = FusionDecision(
            status="SESUAI",
            reason="Seluruh atribut wajib terpenuhi",
            mode="fusion",
            rule_status="SESUAI",
            prob_sesuai=0.94,
            prob_tidak_sesuai=0.06,
            active_violations=(),
            missing_required=(),
        )
        person = PersonAnalysis(
            track_id=1,
            bbox=(10.0, 10.0, float(w - 10), float(h - 20)),
            confidence=0.96,
            full_body_visible=True,
            attributes=[Detection(1, "jas_almamater", 0.91, (20.0, 40.0, 120.0, 180.0))],
            raw_decision=decision,
            smoothed_status="SESUAI",
        )
        return FrameResult(
            frame_width=w,
            frame_height=h,
            persons=[person],
            unassigned_attributes=[],
            timing=FrameTiming(3.1, 4.2, 0.0, 0.2, 7.5, 133.3),
        )


def test_fastapi_rest_and_websocket_endpoints(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("ATTIRE_PROJECT_ROOT", str(tmp_path))
    stub = _StubPipeline()
    app = create_app(pipeline_override=stub, warmup=False)

    dummy_bgr = np.full((240, 320, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", dummy_bgr)
    assert ok
    jpeg_bytes = encoded.tobytes()

    with TestClient(app) as client:
        # 1. Single-page HTML + static offline CSS & JS
        html_res = client.get("/")
        assert html_res.status_code == 200
        assert "Sistem Deteksi Etika Berpakaian Mahasiswa" in html_res.text
        assert client.get("/static/styles.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200

        # 2. Health & Config
        health = client.get("/api/health").json()
        assert health["status"] == "ok"
        assert health["attribute_detector"] == "yolo12n_attire_v0.pt"

        cfg = client.get("/api/config").json()
        assert len(cfg["classes"]) == 6
        assert cfg["classes"][3]["name"] == "bawahan_panjang"

        # 3. Single image inference
        inf_res = client.post(
            "/api/infer-image?conf=0.45",
            files={"file": ("sample.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert inf_res.status_code == 200, inf_res.text
        body = inf_res.json()
        assert body["persons"][0]["status"] == "SESUAI"
        assert body["persons"][0]["attributes"][0]["class_name"] == "jas_almamater"

        # 4. Capture & Export CSV/JSON
        b64_str = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()
        cap_res = client.post("/api/capture", json={"image_base64": b64_str, "analysis": body})
        assert cap_res.status_code == 200
        assert Path(cap_res.json()["saved_image"]).is_file()

        csv_res = client.post(
            "/api/sessions/export",
            json={
                "format": "csv",
                "records": [
                    {"timestamp": "2026-11-22T18:00:00Z", "track_id": 1, "status": "SESUAI"}
                ],
            },
        )
        assert csv_res.status_code == 200
        assert "SESUAI" in csv_res.text

        # 5. WebSocket real-time inference + live config + reset
        with client.websocket_connect("/ws/infer") as ws:
            ws.send_json({"type": "config", "conf": 0.40, "track": True})
            ack = ws.receive_json()
            assert ack["type"] == "config_ack"
            assert ack["conf"] == 0.40

            ws.send_bytes(jpeg_bytes)
            ws_res = ws.receive_json()
            assert ws_res["persons"][0]["status"] == "SESUAI"
            assert ws_res["active_conf"] == 0.40

            ws.send_json({"type": "reset"})
            assert ws.receive_json()["type"] == "reset_ack"
            assert stub.reset_called >= 1
