"""Resolusi path standar proyek."""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Kembalikan root direktori repositori proyek."""
    env_root = os.environ.get("ATTIRE_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = get_project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def resolve_project_path(path: str | Path, root: Path | None = None) -> Path:
    """Ubah path relatif terhadap root proyek menjadi path absolut."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    base = root if root is not None else get_project_root()
    return (base / candidate).resolve()


def configure_runtime_env(root: Path | None = None) -> None:
    """Arahkan direktori cache Ultralytics/HuggingFace/Torch ke dalam proyek (self-contained)."""
    base = root if root is not None else get_project_root()
    ultralytics_dir = base / "outputs" / ".ultralytics"
    mpl_dir = base / "outputs" / ".matplotlib"
    hf_dir = base / "models" / "pretrained" / ".hf_cache"
    torch_dir = base / "models" / "pretrained" / ".torch_cache"
    for d in (ultralytics_dir, mpl_dir, hf_dir, torch_dir):
        d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("HF_HOME", str(hf_dir))
    os.environ.setdefault("TORCH_HOME", str(torch_dir))
    if any((hf_dir / "hub").glob("models--*")):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
