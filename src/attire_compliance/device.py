"""Resolusi device PyTorch/Ultralytics dengan fallback aman ke CPU bila CUDA tidak tersedia."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch


def resolve_device(requested: str | int | None = "auto") -> tuple[int | str, bool]:
    """Tentukan device inferensi/training yang valid untuk Ultralytics & PyTorch.

    Mengembalikan `(ultralytics_device, use_cuda)`:
    - Jika `requested` adalah `"cpu"` → `("cpu", False)`
    - Jika `requested` adalah `"mps"` dan MPS tersedia → `("mps", False)`
    - Jika `torch.cuda.is_available()` bernilai True:
      - `"auto"` / `"cuda"` / `"cuda:0"` / `"0"` / `0` / `None` → `(0, True)`
      - string/indeks CUDA lain (mis. `"1"`, `"cuda:1"`, `"0,1"`) → `(normalized, True)`
    - Jika CUDA tidak tersedia → fallback otomatis ke `("cpu", False)` agar tidak crash.
    """
    raw = "auto" if requested is None else str(requested).strip().lower()
    if raw == "cpu":
        return "cpu", False

    if raw == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps", False
        return "cpu", False

    if not torch.cuda.is_available():
        return "cpu", False

    if raw in ("", "auto", "cuda", "cuda:0", "0"):
        return 0, True
    if raw.startswith("cuda:"):
        idx = raw.split(":", 1)[1]
        return (int(idx) if idx.isdigit() else raw), True
    if raw.isdigit():
        return int(raw), True
    return raw, True


def resolve_torch_device(requested: str | int | None = "auto") -> tuple[torch.device, bool]:
    """Kembalikan `(torch.device, use_cuda)` yang aman untuk model PyTorch/timm."""
    dev, use_cuda = resolve_device(requested)
    if use_cuda:
        if isinstance(dev, int):
            return torch.device(f"cuda:{dev}"), True
        first = str(dev).split(",", 1)[0].strip()
        if first.isdigit():
            return torch.device(f"cuda:{first}"), True
        return torch.device(first if first.startswith("cuda") else "cuda:0"), True
    return torch.device(str(dev)), False


def autocast_context(enabled: bool) -> AbstractContextManager[Any]:
    """Kembalikan `torch.amp.autocast('cuda')` hanya bila CUDA AMP aktif; selain itu no-op."""
    if enabled and torch.cuda.is_available():
        return torch.amp.autocast("cuda", enabled=True)
    return nullcontext()
