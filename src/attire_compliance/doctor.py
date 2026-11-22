"""Diagnostik environment, konfigurasi, GPU CUDA, dan smoke-test YOLOv12 FP16."""

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from attire_compliance.config import load_all_configs
from attire_compliance.paths import configure_runtime_env, resolve_project_path


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_python_version() -> CheckResult:
    major, minor, micro = sys.version_info[:3]
    ok = (3, 11) <= (major, minor) < (3, 13)
    status_hint = "didukung (3.11–3.12)" if ok else "wajib Python 3.11 atau 3.12"
    return CheckResult(
        name="Python",
        ok=ok,
        detail=f"{major}.{minor}.{micro} ({status_hint})",
    )


def check_configs(configs_dir: Path | None = None) -> CheckResult:
    try:
        cfgs = load_all_configs(configs_dir)
    except Exception as exc:
        return CheckResult(name="Konfigurasi YAML", ok=False, detail=str(exc))
    class_list = ", ".join(cfgs.classes.names)
    return CheckResult(
        name="Konfigurasi YAML",
        ok=True,
        detail=f"5 file valid; 6 kelas terkunci: {class_list}",
    )


def check_dependencies() -> CheckResult:
    configure_runtime_env()
    modules = [
        "torch",
        "torchvision",
        "ultralytics",
        "timm",
        "cv2",
        "fastapi",
        "pydantic",
        "sklearn",
    ]
    loaded: list[str] = []
    missing: list[str] = []
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "ok")
            loaded.append(f"{mod_name}={version}")
        except Exception:
            missing.append(mod_name)

    if missing:
        return CheckResult(
            name="Dependensi inti",
            ok=False,
            detail=f"Gagal import: {', '.join(missing)}",
        )
    return CheckResult(name="Dependensi inti", ok=True, detail=", ".join(loaded))


def check_cuda() -> CheckResult:
    try:
        import torch
    except Exception as exc:
        return CheckResult(name="CUDA & GPU", ok=False, detail=f"PyTorch gagal dimuat: {exc}")

    if not torch.cuda.is_available():
        return CheckResult(
            name="CUDA & GPU",
            ok=False,
            detail=f"torch.cuda.is_available() = False (torch={torch.__version__})",
        )

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    vram_gib = props.total_memory / (1024**3)
    cudnn_ver = torch.backends.cudnn.version()
    return CheckResult(
        name="CUDA & GPU",
        ok=True,
        detail=(
            f"{props.name} | VRAM {vram_gib:.2f} GiB | "
            f"PyTorch CUDA {torch.version.cuda} | cuDNN {cudnn_ver}"
        ),
    )


def check_yolo_fp16_inference(configs_dir: Path | None = None) -> CheckResult:
    configure_runtime_env()
    try:
        import numpy as np
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        return CheckResult(name="YOLOv12 FP16 smoke test", ok=False, detail=str(exc))

    if not torch.cuda.is_available():
        return CheckResult(
            name="YOLOv12 FP16 smoke test",
            ok=False,
            detail="Dilewati karena CUDA tidak tersedia",
        )

    try:
        cfgs = load_all_configs(configs_dir)
        weights_path = resolve_project_path(cfgs.app.models.person_detector)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(weights_path))
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)

        with torch.inference_mode():
            # Warmup 1x lalu ukur 1x inference FP16
            model.predict(dummy, imgsz=640, device=0, quantize="fp16", verbose=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = model.predict(dummy, imgsz=640, device=0, quantize="fp16", verbose=False)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        ok = len(results) == 1
        return CheckResult(
            name="YOLOv12 FP16 smoke test",
            ok=ok,
            detail=f"{weights_path.name} @ 640 FP16 selesai dalam {elapsed_ms:.1f} ms",
        )
    except Exception as exc:
        return CheckResult(name="YOLOv12 FP16 smoke test", ok=False, detail=str(exc))


def run_doctor(
    *,
    skip_inference: bool = False,
    configs_dir: Path | None = None,
) -> list[CheckResult]:
    """Jalankan seluruh pemeriksaan diagnostik environment."""
    results = [
        check_python_version(),
        check_configs(configs_dir),
        check_dependencies(),
        check_cuda(),
    ]
    if not skip_inference:
        results.append(check_yolo_fp16_inference(configs_dir))
    return results
