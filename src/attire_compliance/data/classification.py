"""Pembangunan dataset crop person full-body untuk pelatihan EfficientNetV2-S."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import torch

from attire_compliance.classification.model import crop_person_with_padding
from attire_compliance.config import load_all_configs
from attire_compliance.data.merge import IMAGE_EXTS, parse_yolo_label_file
from attire_compliance.detection.predictor import Detection
from attire_compliance.device import resolve_device
from attire_compliance.paths import configure_runtime_env, resolve_project_path
from attire_compliance.pipeline.association import (
    PersonBox,
    associate_attributes_to_persons,
    is_full_body_visible,
)
from attire_compliance.pipeline.rules import evaluate_rules


def _yolo_to_detections(
    labels: tuple[tuple[int, float, float, float, float], ...],
    img_w: int,
    img_h: int,
    id_to_name: dict[int, str],
) -> list[Detection]:
    dets: list[Detection] = []
    for cls_id, xc, yc, bw, bh in labels:
        x1 = (xc - bw * 0.5) * img_w
        y1 = (yc - bh * 0.5) * img_h
        x2 = (xc + bw * 0.5) * img_w
        y2 = (yc + bh * 0.5) * img_h
        dets.append(
            Detection(
                class_id=cls_id,
                class_name=id_to_name[cls_id],
                confidence=1.0,
                bbox=(x1, y1, x2, y2),
            )
        )
    return dets


def _load_fashionpedia_raw_categories(fp_dir: Path) -> dict[str, set[str]]:
    """Muat daftar kategori mentah Fashionpedia (mis. `shorts`) untuk pre-label `tidak_sesuai`."""
    mapping: dict[str, set[str]] = {}
    meta_dir = fp_dir / "metadata"
    if not meta_dir.is_dir():
        return mapping
    for jsonl in meta_dir.glob("*.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[str(row["image_id"])] = set(row.get("raw_categories", []))
    return mapping


def prelabel_crop_from_gt(
    attr_names: set[str],
    *,
    almamater_overrides_tshirt: bool = True,
    has_shorts_meta: bool = False,
) -> tuple[str | None, str]:
    """Tentukan pre-label `sesuai` / `tidak_sesuai` hanya untuk crop dengan bukti GT yang jelas.

    Mengembalikan `(None, alasan)` bila anotasi atribut pada orang tersebut parsial/ambigu
    sehingga tidak mencemari training classifier.
    """
    rule_eval = evaluate_rules(
        attr_names,
        full_body_visible=True,
        almamater_overrides_tshirt=almamater_overrides_tshirt,
        allow_incomplete=False,
    )
    if rule_eval.active_violations:
        return "tidak_sesuai", rule_eval.reason
    if rule_eval.top_ok and rule_eval.bottom_ok and rule_eval.foot_ok:
        return "sesuai", rule_eval.reason
    if has_shorts_meta and not rule_eval.bottom_ok and (rule_eval.top_ok or rule_eval.foot_ok):
        return "tidak_sesuai", "Celana pendek (shorts) terdeteksi pada metadata sumber"
    return None, "atribut_tidak_lengkap_atau_ambigu"


def build_classification_dataset(
    *,
    version: str = "v1",
    detection_version: str | None = None,
    person_conf: float = 0.45,
    min_person_height: int = 120,
    min_aspect_ratio: float = 1.15,
    pad_ratio: float | None = None,
    device: str = "0",
    configs_dir: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Ekstrak crop person full-body dari dataset deteksi ke dataset klasifikasi 2-kelas.

    Mempertahankan split `train/val/test` persis seperti dataset deteksi agar zero leakage,
    dan mengekspor `review_queue.csv` agar pre-label mudah diverifikasi manusia.
    """
    configure_runtime_env(project_root)
    from ultralytics import YOLO

    cfgs = load_all_configs(configs_dir)
    classes_cfg = cfgs.classes
    cls_cfg = cfgs.classifier
    app_cfg = cfgs.app

    det_ver = detection_version or version
    det_dir = resolve_project_path(f"data/processed/detection/{det_ver}", project_root)
    if not det_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset deteksi `{det_ver}` tidak ditemukan di {det_dir}. "
            "Jalankan `attire data build-detection` terlebih dulu."
        )

    out_dir = resolve_project_path(f"data/processed/classification/{version}", project_root)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split_name in ("train", "val", "test"):
        for label_name in cls_cfg.class_names:
            (out_dir / split_name / label_name).mkdir(parents=True, exist_ok=True)

    person_weights = resolve_project_path(app_cfg.models.person_detector, project_root)
    person_model = YOLO(str(person_weights))
    run_device, _ = resolve_device(device)

    fp_meta = _load_fashionpedia_raw_categories(
        resolve_project_path(cfgs.sources.fashionpedia.output_dir, project_root)
    )
    pad = pad_ratio if pad_ratio is not None else cls_cfg.pad_ratio
    id_to_name = classes_cfg.id_to_name
    num_classes = len(classes_cfg.names)

    split_counts: dict[str, dict[str, int]] = {}
    skipped_reasons: Counter[str] = Counter()
    review_rows: list[dict[str, Any]] = []
    total_saved = 0

    for split_name in ("train", "val", "test"):
        img_dir = det_dir / "images" / split_name
        lbl_dir = det_dir / "labels" / split_name
        counts = Counter[str]()

        if img_dir.is_dir():
            for img_path in sorted(img_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                labels, _ = parse_yolo_label_file(lbl_path, num_classes=num_classes)
                if not labels:
                    continue

                image_bgr = cv2.imread(str(img_path))
                if image_bgr is None:
                    continue
                img_h, img_w = image_bgr.shape[:2]
                gt_dets = _yolo_to_detections(labels, img_w, img_h, id_to_name)

                with torch.inference_mode():
                    res = person_model.predict(
                        image_bgr,
                        classes=[0],
                        conf=person_conf,
                        imgsz=640,
                        device=run_device,
                        verbose=False,
                    )
                boxes = res[0].boxes if res else None
                if boxes is None or len(boxes) == 0:
                    skipped_reasons["no_person_detected"] += 1
                    continue

                persons = [
                    PersonBox(
                        track_id=idx,
                        bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        confidence=float(c),
                    )
                    for idx, (b, c) in enumerate(
                        zip(
                            boxes.xyxy.detach().cpu().numpy(),
                            boxes.conf.detach().cpu().numpy(),
                            strict=True,
                        )
                    )
                ]
                per_person, _ = associate_attributes_to_persons(
                    persons, gt_dets, min_ioa=app_cfg.thresholds.association_min_ioa
                )

                # Cek metadata Fashionpedia bila stem mengandung `fp_train_` / `fp_val_`
                has_shorts = False
                for key_part in ("fp_train_", "fp_val_"):
                    pos = img_path.stem.find(key_part)
                    if pos != -1:
                        raw_cats = fp_meta.get(img_path.stem[pos:], set())
                        has_shorts = "shorts" in raw_cats
                        break

                for person in persons:
                    x1, y1, x2, y2 = person.bbox
                    pw, ph = max(1.0, x2 - x1), max(1.0, y2 - y1)
                    if ph < min_person_height or (ph / pw) < min_aspect_ratio:
                        skipped_reasons["crop_too_small_or_not_portrait"] += 1
                        continue
                    if not is_full_body_visible(
                        person.bbox,
                        img_h,
                        bottom_edge_margin_px=max(2, app_cfg.fusion.bottom_edge_margin_px // 2),
                    ):
                        # Izinkan tetap masuk jika alas kaki (sepatu/sandal) benar-benar ada di GT
                        attr_set_tmp = {d.class_name for d in per_person[person.track_id]}
                        if not (attr_set_tmp & {"sepatu", "sandal"}):
                            skipped_reasons["feet_clipped_at_bottom"] += 1
                            continue

                    attr_names = {d.class_name for d in per_person[person.track_id]}
                    if not attr_names:
                        skipped_reasons["no_gt_attributes_associated"] += 1
                        continue

                    prelabel, reason = prelabel_crop_from_gt(
                        attr_names,
                        almamater_overrides_tshirt=app_cfg.fusion.almamater_overrides_tshirt,
                        has_shorts_meta=has_shorts,
                    )
                    if prelabel is None:
                        skipped_reasons[reason] += 1
                        continue

                    crop_pil = crop_person_with_padding(image_bgr, person.bbox, pad_ratio=pad)
                    crop_name = f"{img_path.stem}_p{person.track_id:02d}.jpg"
                    dst_path = out_dir / split_name / prelabel / crop_name
                    crop_pil.save(dst_path, format="JPEG", quality=92)
                    counts[prelabel] += 1
                    total_saved += 1
                    review_rows.append(
                        {
                            "split": split_name,
                            "prelabel": prelabel,
                            "file": f"{split_name}/{prelabel}/{crop_name}",
                            "source_image": img_path.name,
                            "gt_attributes": ",".join(sorted(attr_names)),
                            "reason": reason,
                            "verified": "false",
                        }
                    )

        split_counts[split_name] = {name: int(counts.get(name, 0)) for name in cls_cfg.class_names}

    with (out_dir / "review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "prelabel",
            "file",
            "source_image",
            "gt_attributes",
            "reason",
            "verified",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    totals = {
        name: sum(split_counts[s].get(name, 0) for s in ("train", "val", "test"))
        for name in cls_cfg.class_names
    }
    manifest: dict[str, Any] = {
        "version": version,
        "detection_version": det_ver,
        "output_dir": str(out_dir.resolve()),
        "total_crops": total_saved,
        "splits": split_counts,
        "total_by_class": totals,
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
        "review_queue_csv": str((out_dir / "review_queue.csv").resolve()),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
