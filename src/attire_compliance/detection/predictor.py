"""Wrapper inferensi detektor objek YOLOv12 yang dipakai lintas prelabel, pipeline, dan app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from attire_compliance.device import resolve_device
from attire_compliance.paths import configure_runtime_env


@dataclass(frozen=True)
class Detection:
    """Satu objek terdeteksi dari model YOLO."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) piksel


class AttributeDetector:
    """Detektor 6 atribut pakaian berbasis Ultralytics YOLOv12."""

    def __init__(
        self,
        weights_path: str | Path,
        class_names: dict[int, str],
        *,
        device: str = "0",
        half: bool = True,
        imgsz: int = 640,
    ) -> None:
        configure_runtime_env()
        from ultralytics import YOLO

        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"Bobot detektor tidak ditemukan: {path}")

        self.weights_path = path
        self.class_names = dict(class_names)
        self.device, self.use_cuda = resolve_device(device)
        self.quantize = "fp16" if (half and self.use_cuda) else None
        self.imgsz = imgsz
        self.model: Any = YOLO(str(path))

    def predict(
        self,
        image: np.ndarray | str | Path,
        *,
        conf: float = 0.45,
        iou: float = 0.50,
        per_class_thresholds: dict[str, float] | None = None,
    ) -> list[Detection]:
        """Jalankan inferensi pada satu frame/citra dan kembalikan daftar `Detection`."""
        return self.predict_batch(
            [image], conf=conf, iou=iou, per_class_thresholds=per_class_thresholds
        )[0]

    def predict_batch(
        self,
        images: list[np.ndarray | str | Path],
        *,
        conf: float = 0.45,
        iou: float = 0.50,
        per_class_thresholds: dict[str, float] | None = None,
    ) -> list[list[Detection]]:
        """Jalankan inferensi ter-batch pada daftar citra/crop dalam 1 forward pass GPU."""
        if not images:
            return []
        min_conf = conf
        if per_class_thresholds:
            min_conf = min(conf, *per_class_thresholds.values())

        kwargs: dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": min_conf,
            "iou": iou,
            "device": self.device,
            "verbose": False,
        }
        if self.quantize is not None:
            kwargs["quantize"] = self.quantize

        with torch.inference_mode():
            results = self.model.predict(images, **kwargs)

        batch_out: list[list[Detection]] = []
        for res in results:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                batch_out.append([])
                continue
            xyxy_arr = boxes.xyxy.detach().cpu().numpy()
            conf_arr = boxes.conf.detach().cpu().numpy()
            cls_arr = boxes.cls.detach().cpu().numpy().astype(int)
            detections: list[Detection] = []
            for coords, score, cls_id in zip(xyxy_arr, conf_arr, cls_arr, strict=True):
                cls_name = self.class_names.get(int(cls_id), f"class_{int(cls_id)}")
                threshold = (
                    per_class_thresholds.get(cls_name, conf) if per_class_thresholds else conf
                )
                if float(score) < threshold:
                    continue
                x1, y1, x2, y2 = (float(v) for v in coords)
                detections.append(
                    Detection(
                        class_id=int(cls_id),
                        class_name=cls_name,
                        confidence=float(score),
                        bbox=(x1, y1, x2, y2),
                    )
                )
            batch_out.append(detections)
        return batch_out
