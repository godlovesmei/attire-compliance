"""Arsitektur, preprocessing letterbox, transformasi, dan checkpoint EfficientNetV2-S."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from attire_compliance.paths import configure_runtime_env


class LetterboxResize:
    """Resize citra dengan menjaga rasio aspek lalu pad menjadi persegi `size x size`."""

    def __init__(self, size: int = 224, fill_color: tuple[int, int, int] = (114, 114, 114)) -> None:
        self.size = size
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        return letterbox_rgb(img, size=self.size, fill_color=self.fill_color)


def letterbox_rgb(
    img: Image.Image,
    size: int = 224,
    fill_color: tuple[int, int, int] = (114, 114, 114),
) -> Image.Image:
    """Jaga rasio aspek crop tubuh portrait agar tidak terdistorsi saat di-resize ke persegi."""
    rgb = img.convert("RGB")
    width, height = rgb.size
    if width <= 0 or height <= 0:
        return Image.new("RGB", (size, size), fill_color)
    scale = min(size / width, size / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill_color)
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def crop_person_with_padding(
    image_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad_ratio: float = 0.10,
) -> Image.Image:
    """Crop ROI person dari frame BGR dengan padding proporsional dan kembalikan RGB PIL."""
    img_h, img_w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * pad_ratio
    py = bh * pad_ratio
    cx1 = max(0, int(round(x1 - px)))
    cy1 = max(0, int(round(y1 - py)))
    cx2 = min(img_w, int(round(x2 + px)))
    cy2 = min(img_h, int(round(y2 + py)))
    if cx2 <= cx1 or cy2 <= cy1:
        return Image.new("RGB", (224, 224), (114, 114, 114))
    crop_rgb = cv2.cvtColor(image_bgr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def create_classifier_model(
    backbone: str,
    *,
    num_classes: int = 2,
    pretrained: bool = True,
    drop_rate: float = 0.2,
) -> tuple[nn.Module, dict[str, Any]]:
    """Bangun model `timm` EfficientNetV2 beserta metadata normalisasi resmi."""
    configure_runtime_env()
    import timm

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    data_cfg = timm.data.resolve_model_data_config(model)
    mean = tuple(float(v) for v in data_cfg.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(v) for v in data_cfg.get("std", (0.229, 0.224, 0.225)))
    return model, {"mean": mean, "std": std}


def build_classifier_transforms(
    imgsz: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    *,
    is_train: bool,
    use_letterbox: bool = True,
) -> transforms.Compose:
    """Transformasi train/eval konsisten dengan normalisasi ImageNet/timm."""
    resize_op: Any = (
        LetterboxResize(imgsz)
        if use_letterbox
        else transforms.Resize((imgsz, imgsz), interpolation=transforms.InterpolationMode.BICUBIC)
    )
    if is_train:
        return transforms.Compose(
            [
                resize_op,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=6,
                            translate=(0.04, 0.04),
                            scale=(0.90, 1.05),
                            fill=(114, 114, 114),
                        )
                    ],
                    p=0.6,
                ),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            resize_op,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def save_classifier_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    backbone: str,
    class_names: list[str],
    imgsz: int,
    letterbox: bool,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    metrics: dict[str, Any],
) -> None:
    """Simpan state_dict beserta metadata lengkap agar inferensi 100% self-contained."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "backbone": backbone,
        "class_names": list(class_names),
        "imgsz": int(imgsz),
        "letterbox": bool(letterbox),
        "mean": list(mean),
        "std": list(std),
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_classifier_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Muat model classifier & metadata dari file checkpoint `.pt`."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint classifier tidak ditemukan: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    class_names: list[str] = list(payload["class_names"])
    model, _ = create_classifier_model(
        backbone=str(payload["backbone"]),
        num_classes=len(class_names),
        pretrained=False,
        drop_rate=0.0,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = {
        "backbone": str(payload["backbone"]),
        "class_names": class_names,
        "imgsz": int(payload["imgsz"]),
        "letterbox": bool(payload.get("letterbox", True)),
        "mean": tuple(float(v) for v in payload["mean"]),
        "std": tuple(float(v) for v in payload["std"]),
        "metrics": payload.get("metrics", {}),
    }
    return model, meta
