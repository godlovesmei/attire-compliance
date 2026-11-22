"""Wrapper inferensi EfficientNetV2-S untuk klasifikasi kepatuhan berpakaian."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from attire_compliance.classification.model import (
    build_classifier_transforms,
    load_classifier_checkpoint,
)


@dataclass(frozen=True)
class CompliancePrediction:
    """Hasil prediksi biner EfficientNetV2-S untuk satu crop person."""

    label: str
    prob_sesuai: float
    prob_tidak_sesuai: float


class ComplianceClassifier:
    """Classifier EfficientNetV2-S yang dimuat dari checkpoint `.pt` self-contained."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        half: bool = True,
    ) -> None:
        path = Path(checkpoint_path)
        self.use_cuda = torch.cuda.is_available() and device != "cpu"
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.half = bool(half and self.use_cuda)

        model, meta = load_classifier_checkpoint(path, map_location=self.device)
        self.model = model.to(self.device).eval()
        self.meta = meta
        self.class_names: list[str] = meta["class_names"]
        self.idx_sesuai = self.class_names.index("sesuai")
        self.idx_tidak = self.class_names.index("tidak_sesuai")
        self.transform = build_classifier_transforms(
            imgsz=meta["imgsz"],
            mean=meta["mean"],
            std=meta["std"],
            is_train=False,
            use_letterbox=meta["letterbox"],
        )

    def predict_crops(self, crops: list[Image.Image]) -> list[CompliancePrediction]:
        """Klasifikasikan satu batch crop person RGB secara paralel di GPU."""
        if not crops:
            return []
        tensors = [self.transform(crop.convert("RGB")) for crop in crops]
        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)

        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=self.half):
            logits = self.model(batch)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        predictions: list[CompliancePrediction] = []
        for row in probs:
            p_sesuai = float(row[self.idx_sesuai])
            p_tidak = float(row[self.idx_tidak])
            label = "sesuai" if p_sesuai >= p_tidak else "tidak_sesuai"
            predictions.append(
                CompliancePrediction(
                    label=label,
                    prob_sesuai=p_sesuai,
                    prob_tidak_sesuai=p_tidak,
                )
            )
        return predictions
