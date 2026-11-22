"""Pre-annotation otomatis frame lokal menggunakan detektor v0 (format YOLO siap review)."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from attire_compliance.config import load_classes_config
from attire_compliance.data.geometry import box_area, clip_bbox, intersection_area, xyxy_to_yolo
from attire_compliance.detection.predictor import AttributeDetector, Detection
from attire_compliance.paths import configure_runtime_env, resolve_project_path

# Warna BGR kontras per kelas untuk preview visual
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (60, 180, 75),   # pakaian_berkerah (hijau)
    1: (200, 130, 0),   # jas_almamater (biru)
    2: (48, 48, 230),   # kaos_oblong (merah)
    3: (180, 119, 31),  # bawahan_panjang (biru tua)
    4: (25, 225, 255),  # sepatu (kuning)
    5: (240, 50, 230),  # sandal (magenta)
}


def _draw_preview(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    canvas = image_bgr.copy()
    thickness = max(2, min(canvas.shape[:2]) // 300)
    font_scale = max(0.5, min(canvas.shape[:2]) / 1200.0)
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        color = CLASS_COLORS_BGR.get(det.class_id, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        ty1 = max(0, y1 - th - 8)
        cv2.rectangle(canvas, (x1, ty1), (x1 + tw + 6, ty1 + th + 6), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, ty1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def nms_detections(detections: list[Detection], iou_threshold: float = 0.50) -> list[Detection]:
    """Class-wise Non-Maximum Suppression untuk menggabungkan deteksi dari beberapa crop person."""
    by_class: dict[int, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    kept: list[Detection] = []
    for cls_dets in by_class.values():
        ordered = sorted(cls_dets, key=lambda d: d.confidence, reverse=True)
        selected: list[Detection] = []
        for cand in ordered:
            if all(_box_iou(cand.bbox, prev.bbox) < iou_threshold for prev in selected):
                selected.append(cand)
        kept.extend(selected)
    return sorted(kept, key=lambda d: d.confidence, reverse=True)


def _predict_with_person_crops(
    image_bgr: np.ndarray,
    detector: AttributeDetector,
    person_model: Any | None,
    *,
    conf: float,
    iou: float,
    person_conf: float = 0.35,
    pad_ratio: float = 0.08,
) -> list[Detection]:
    """Jalankan deteksi atribut per crop `person` (Stage 0) + fallback full-frame."""
    img_h, img_w = image_bgr.shape[:2]
    all_dets: list[Detection] = []

    if person_model is not None:
        import torch

        with torch.inference_mode():
            res = person_model.predict(
                image_bgr,
                classes=[0],
                conf=person_conf,
                imgsz=640,
                device=detector.device,
                verbose=False,
            )
        boxes = res[0].boxes if res else None
        if boxes is not None and len(boxes) > 0:
            for xyxy in boxes.xyxy.detach().cpu().numpy():
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                pw, ph = (x2 - x1) * pad_ratio, (y2 - y1) * pad_ratio
                cx1 = max(0, int(round(x1 - pw)))
                cy1 = max(0, int(round(y1 - ph)))
                cx2 = min(img_w, int(round(x2 + pw)))
                cy2 = min(img_h, int(round(y2 + ph)))
                if (cx2 - cx1) < 32 or (cy2 - cy1) < 32:
                    continue
                crop = image_bgr[cy1:cy2, cx1:cx2]
                for det in detector.predict(crop, conf=conf, iou=iou):
                    dx1, dy1, dx2, dy2 = det.bbox
                    all_dets.append(
                        Detection(
                            class_id=det.class_id,
                            class_name=det.class_name,
                            confidence=det.confidence,
                            bbox=(dx1 + cx1, dy1 + cy1, dx2 + cx1, dy2 + cy1),
                        )
                    )

    # Tambahkan juga full-frame bila tidak ada person terdeteksi atau untuk objek besar
    if not all_dets:
        all_dets.extend(detector.predict(image_bgr, conf=conf, iou=iou))

    return nms_detections(all_dets, iou_threshold=iou)


def prelabel_local_frames(
    *,
    weights_path: Path,
    frames_dir: Path | None = None,
    output_dir: Path | None = None,
    conf: float = 0.25,
    iou: float = 0.50,
    max_previews: int = 12,
    use_person_crops: bool = True,
    person_weights: Path | None = None,
    device: str = "0",
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """Hasilkan pre-label YOLO otomatis pada seluruh frame di `data/interim/frames/`."""
    classes_cfg = load_classes_config(configs_dir)
    src_frames = (
        frames_dir if frames_dir is not None else resolve_project_path("data/interim/frames")
    )
    out_root = (
        output_dir
        if output_dir is not None
        else resolve_project_path("data/interim/annotations")
    )
    prelabels_dir = out_root / "prelabels"
    previews_dir = out_root / "previews"

    for directory in (prelabels_dir, previews_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = src_frames / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_rel_paths = [Path(rec["file"]) for rec in manifest.get("frames", [])]
    else:
        frame_rel_paths = sorted(
            p.relative_to(src_frames)
            for p in src_frames.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    detector = AttributeDetector(
        weights_path=weights_path,
        class_names=classes_cfg.id_to_name,
        device=device,
        half=True,
    )

    person_model: Any | None = None
    if use_person_crops:
        configure_runtime_env()
        from ultralytics import YOLO

        p_weights = (
            person_weights
            if person_weights is not None
            else resolve_project_path("models/pretrained/yolo12n.pt")
        )
        if p_weights.is_file():
            person_model = YOLO(str(p_weights))

    total_frames = len(frame_rel_paths)
    preview_indices: set[int] = set()
    if total_frames > 0 and max_previews > 0:
        count = min(max_previews, total_frames)
        preview_indices = {
            int(i) for i in np.linspace(0, total_frames - 1, count).round().astype(int)
        }

    class_counts: Counter[str] = Counter()
    conf_sums: Counter[str] = Counter()
    by_batch_boxes: dict[str, Counter[str]] = {}
    zero_det_frames: list[str] = []
    preview_files: list[str] = []

    for idx, rel_path in enumerate(frame_rel_paths):
        img_path = src_frames / rel_path
        if not img_path.is_file():
            continue
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        img_h, img_w = image_bgr.shape[:2]

        detections = _predict_with_person_crops(
            image_bgr, detector, person_model, conf=conf, iou=iou
        )
        batch = rel_path.parts[0] if len(rel_path.parts) > 1 else "default"
        batch_counter = by_batch_boxes.setdefault(batch, Counter())

        yolo_lines: list[str] = []
        for det in detections:
            clipped = clip_bbox(det.bbox, img_w, img_h)
            if clipped is None:
                continue
            xc, yc, bw, bh = xyxy_to_yolo(clipped, img_w, img_h)
            yolo_lines.append(f"{det.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            class_counts[det.class_name] += 1
            conf_sums[det.class_name] += det.confidence
            batch_counter[det.class_name] += 1

        if not yolo_lines:
            zero_det_frames.append(rel_path.as_posix())

        dst_lbl = (prelabels_dir / rel_path).with_suffix(".txt")
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.write_text(("\n".join(yolo_lines) + "\n") if yolo_lines else "", encoding="utf-8")

        if idx in preview_indices:
            preview_img = _draw_preview(image_bgr, detections)
            preview_name = rel_path.as_posix().replace("/", "__")
            preview_path = previews_dir / preview_name
            cv2.imwrite(str(preview_path), preview_img)
            preview_files.append(preview_path.name)

    (prelabels_dir / "classes.txt").write_text(
        "\n".join(classes_cfg.names) + "\n", encoding="utf-8"
    )

    mean_conf = {
        name: round(conf_sums[name] / class_counts[name], 4) if class_counts[name] > 0 else 0.0
        for name in classes_cfg.names
    }
    report: dict[str, Any] = {
        "weights_path": str(Path(weights_path).resolve()),
        "use_person_crops": person_model is not None,
        "frames_dir": str(src_frames.resolve()),
        "prelabels_dir": str(prelabels_dir.resolve()),
        "previews_dir": str(previews_dir.resolve()),
        "conf": conf,
        "iou": iou,
        "total_frames": total_frames,
        "frames_with_detections": total_frames - len(zero_det_frames),
        "zero_detection_frames": zero_det_frames,
        "instances": {name: int(class_counts.get(name, 0)) for name in classes_cfg.names},
        "mean_confidence": mean_conf,
        "by_batch": {
            batch: {name: int(cnt.get(name, 0)) for name in classes_cfg.names}
            for batch, cnt in sorted(by_batch_boxes.items())
        },
        "preview_files": preview_files,
    }
    (out_root / "prelabel_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
