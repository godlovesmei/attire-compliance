"""Orkestrasi Hybrid Pipeline Stage 0–3 (dipakai bersama oleh FastAPI app & evaluasi)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from attire_compliance.classification.model import crop_person_with_padding
from attire_compliance.classification.predictor import ComplianceClassifier, CompliancePrediction
from attire_compliance.config import FusionMode, ProjectConfigs, load_all_configs
from attire_compliance.detection.predictor import AttributeDetector, Detection
from attire_compliance.detection.prelabel import nms_detections
from attire_compliance.paths import configure_runtime_env, resolve_project_path
from attire_compliance.pipeline.association import (
    PersonBox,
    associate_attributes_to_persons,
    is_full_body_visible,
)
from attire_compliance.pipeline.fusion import FusionDecision, fuse_decision
from attire_compliance.pipeline.rules import ComplianceStatus, evaluate_rules
from attire_compliance.pipeline.smoothing import TemporalSmoother


@dataclass(frozen=True)
class PersonAnalysis:
    """Hasil analisis lengkap untuk satu mahasiswa dalam satu frame."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    full_body_visible: bool
    attributes: list[Detection]
    raw_decision: FusionDecision
    smoothed_status: ComplianceStatus


@dataclass(frozen=True)
class FrameTiming:
    person_ms: float
    attribute_ms: float
    classifier_ms: float
    fusion_ms: float
    total_ms: float
    fps: float


@dataclass(frozen=True)
class FrameResult:
    frame_width: int
    frame_height: int
    persons: list[PersonAnalysis]
    unassigned_attributes: list[Detection]
    timing: FrameTiming


class AttireCompliancePipeline:
    """Two-stage hybrid pipeline: YOLOv12 (person + 6 atribut) + EfficientNetV2-S + Fusion."""

    def __init__(
        self,
        *,
        configs: ProjectConfigs | None = None,
        configs_dir: Path | None = None,
        person_weights: Path | None = None,
        attribute_weights: Path | None = None,
        classifier_weights: Path | None = None,
        person_crop_attributes: bool = True,
    ) -> None:
        configure_runtime_env()
        from ultralytics import YOLO

        self.configs = configs if configs is not None else load_all_configs(configs_dir)
        self.app_cfg = self.configs.app
        self.classes_cfg = self.configs.classes
        self.person_crop_attributes = person_crop_attributes

        dev = self.app_cfg.inference.device
        self.use_cuda = torch.cuda.is_available() and dev != "cpu"
        self.device: int | str = 0 if (self.use_cuda and dev in ("cuda", "0")) else dev
        self.half = bool(self.app_cfg.inference.half and self.use_cuda)
        self.quantize = "fp16" if self.half else None

        # Stage 0: YOLOv12 pretrained COCO (kelas person = 0)
        p_path = (
            person_weights
            if person_weights is not None
            else resolve_project_path(self.app_cfg.models.person_detector)
        )
        self.person_weights_path = p_path
        self.person_model: Any = YOLO(str(p_path))

        # Stage 1: YOLOv12 fine-tuned 6 atribut (fallback otomatis ke v0 bila v1 belum ada)
        a_path = (
            attribute_weights
            if attribute_weights is not None
            else resolve_project_path(self.app_cfg.models.attribute_detector)
        )
        if not a_path.is_file():
            fallback_v0 = resolve_project_path("models/detector/yolo12n_attire_v0.pt")
            if fallback_v0.is_file():
                a_path = fallback_v0
        self.attribute_weights_path = a_path if a_path.is_file() else None
        self.attribute_detector: AttributeDetector | None = (
            AttributeDetector(
                weights_path=a_path,
                class_names=self.classes_cfg.id_to_name,
                device=dev,
                half=self.half,
                imgsz=self.app_cfg.inference.imgsz_detector,
            )
            if self.attribute_weights_path is not None
            else None
        )

        # Stage 2: EfficientNetV2-S (opsional saat bobot belum dilatih; fallback rule_only)
        c_path = (
            classifier_weights
            if classifier_weights is not None
            else resolve_project_path(self.app_cfg.models.classifier)
        )
        self.classifier_weights_path = c_path if c_path.is_file() else None
        self.classifier: ComplianceClassifier | None = (
            ComplianceClassifier(c_path, device=dev, half=self.half)
            if self.classifier_weights_path is not None
            else None
        )

        self.smoother = TemporalSmoother(window_size=self.app_cfg.fusion.smoothing_window)

    def warmup(self) -> None:
        """Warmup seluruh model aktif (Stage 0, 1, 2) agar frame pertama langsung cepat."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._detect_persons(dummy, track=False)
        if self.attribute_detector is not None:
            self.attribute_detector.predict(dummy, conf=0.50, iou=0.50)
        if self.classifier is not None:
            from PIL import Image

            self.classifier.predict_crops([Image.new("RGB", (224, 224), (114, 114, 114))])

    def reset_tracking(self) -> None:
        """Reset riwayat smoother dan tracker untuk sesi baru / tombol Reset UI."""
        self.smoother.reset()
        predictor = getattr(self.person_model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            predictor.trackers = None

    def _detect_persons(self, frame_bgr: np.ndarray, *, track: bool) -> list[PersonBox]:
        kwargs: dict[str, Any] = {
            "classes": [0],
            "conf": self.app_cfg.thresholds.person_conf,
            "iou": self.app_cfg.thresholds.iou_nms,
            "imgsz": self.app_cfg.inference.imgsz_detector,
            "device": self.device,
            "verbose": False,
        }
        if self.quantize is not None:
            kwargs["quantize"] = self.quantize

        with torch.inference_mode():
            if track:
                results = self.person_model.track(
                    frame_bgr,
                    persist=True,
                    tracker=self.app_cfg.inference.tracker,
                    **kwargs,
                )
            else:
                results = self.person_model.predict(frame_bgr, **kwargs)

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        ids = (
            boxes.id.detach().cpu().numpy().astype(int).tolist()
            if boxes.id is not None
            else list(range(1, len(xyxy) + 1))
        )

        persons = [
            PersonBox(
                track_id=int(tid),
                bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                confidence=float(score),
            )
            for tid, b, score in zip(ids, xyxy, confs, strict=True)
        ]
        # Urutkan dari kiri ke kanan agar konsisten di UI
        return sorted(persons, key=lambda p: p.bbox[0])

    def _detect_attributes(
        self,
        frame_bgr: np.ndarray,
        persons: list[PersonBox],
        *,
        conf: float,
        iou: float,
        per_class_thresholds: dict[str, float] | None,
    ) -> list[Detection]:
        if self.attribute_detector is None:
            return []

        if not self.person_crop_attributes or not persons:
            return self.attribute_detector.predict(
                frame_bgr, conf=conf, iou=iou, per_class_thresholds=per_class_thresholds
            )

        img_h, img_w = frame_bgr.shape[:2]
        pad = self.app_cfg.inference.person_pad_ratio
        crops: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []
        for person in persons:
            x1, y1, x2, y2 = person.bbox
            pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
            cx1 = max(0, int(round(x1 - pw)))
            cy1 = max(0, int(round(y1 - ph)))
            cx2 = min(img_w, int(round(x2 + pw)))
            cy2 = min(img_h, int(round(y2 + ph)))
            if (cx2 - cx1) < 24 or (cy2 - cy1) < 24:
                continue
            crops.append(frame_bgr[cy1:cy2, cx1:cx2])
            offsets.append((cx1, cy1))

        if not crops:
            return []

        batch_dets = self.attribute_detector.predict_batch(
            crops, conf=conf, iou=iou, per_class_thresholds=per_class_thresholds
        )
        all_dets: list[Detection] = []
        for dets, (ox, oy) in zip(batch_dets, offsets, strict=True):
            for det in dets:
                dx1, dy1, dx2, dy2 = det.bbox
                all_dets.append(
                    Detection(
                        class_id=det.class_id,
                        class_name=det.class_name,
                        confidence=det.confidence,
                        bbox=(dx1 + ox, dy1 + oy, dx2 + ox, dy2 + oy),
                    )
                )
        return nms_detections(all_dets, iou_threshold=iou)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        conf_override: float | None = None,
        fusion_mode_override: FusionMode | None = None,
        track: bool = True,
        use_smoothing: bool = True,
        allow_incomplete: bool = True,
    ) -> FrameResult:
        """Jalankan seluruh pipeline Stage 0–3 pada satu frame BGR."""
        t_start = time.perf_counter()
        img_h, img_w = frame_bgr.shape[:2]

        # Stage 0: Deteksi + tracking person
        t0 = time.perf_counter()
        persons = self._detect_persons(frame_bgr, track=track)
        person_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 1: Deteksi 6 atribut pakaian
        t1 = time.perf_counter()
        default_conf = (
            conf_override
            if conf_override is not None
            else self.app_cfg.thresholds.attribute_conf_default
        )
        per_class = None if conf_override is not None else self.app_cfg.thresholds.per_class
        raw_attrs = self._detect_attributes(
            frame_bgr,
            persons,
            conf=default_conf,
            iou=self.app_cfg.thresholds.iou_nms,
            per_class_thresholds=per_class,
        )
        attribute_ms = (time.perf_counter() - t1) * 1000.0

        # Asosiasi atribut -> person
        per_person_attrs, unassigned_attrs = associate_attributes_to_persons(
            persons, raw_attrs, min_ioa=self.app_cfg.thresholds.association_min_ioa
        )

        # Stage 2: Klasifikasi EfficientNetV2-S (batch semua person dalam 1 forward pass)
        t2 = time.perf_counter()
        clf_preds: list[CompliancePrediction | None] = [None] * len(persons)
        if self.classifier is not None and persons:
            crops = [
                crop_person_with_padding(
                    frame_bgr, p.bbox, pad_ratio=self.app_cfg.inference.person_pad_ratio
                )
                for p in persons
            ]
            clf_preds = list(self.classifier.predict_crops(crops))
        classifier_ms = (time.perf_counter() - t2) * 1000.0

        # Stage 3: Rule engine + Decision Fusion + Temporal Smoothing
        t3 = time.perf_counter()
        if use_smoothing:
            self.smoother.step_frame()

        mode: FusionMode = fusion_mode_override or self.app_cfg.fusion.mode
        analyses: list[PersonAnalysis] = []

        for person, pred in zip(persons, clf_preds, strict=True):
            attrs = per_person_attrs.get(person.track_id, [])
            visible = is_full_body_visible(
                person.bbox,
                img_h,
                bottom_edge_margin_px=self.app_cfg.fusion.bottom_edge_margin_px,
            )
            rule_eval = evaluate_rules(
                [a.class_name for a in attrs],
                full_body_visible=visible,
                almamater_overrides_tshirt=self.app_cfg.fusion.almamater_overrides_tshirt,
                allow_incomplete=allow_incomplete,
            )
            decision = fuse_decision(
                rule_eval,
                prob_sesuai=pred.prob_sesuai if pred is not None else None,
                mode=mode,
                classifier_veto_threshold=self.app_cfg.fusion.classifier_veto_threshold,
                full_body_visible=visible,
                allow_incomplete=allow_incomplete,
            )
            smoothed = (
                self.smoother.update(person.track_id, decision.status)
                if use_smoothing
                else decision.status
            )
            analyses.append(
                PersonAnalysis(
                    track_id=person.track_id,
                    bbox=person.bbox,
                    confidence=person.confidence,
                    full_body_visible=visible,
                    attributes=attrs,
                    raw_decision=decision,
                    smoothed_status=smoothed,
                )
            )
        fusion_ms = (time.perf_counter() - t3) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        return FrameResult(
            frame_width=img_w,
            frame_height=img_h,
            persons=analyses,
            unassigned_attributes=unassigned_attrs,
            timing=FrameTiming(
                person_ms=round(person_ms, 2),
                attribute_ms=round(attribute_ms, 2),
                classifier_ms=round(classifier_ms, 2),
                fusion_ms=round(fusion_ms, 2),
                total_ms=round(total_ms, 2),
                fps=round(fps, 1),
            ),
        )
