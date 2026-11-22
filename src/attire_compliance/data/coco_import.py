"""Penggabungan & konversi ekspor anotasi COCO (Roboflow) ke label YOLO lokal yang tervalidasi."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from attire_compliance.config import load_classes_config
from attire_compliance.detection.predictor import Detection
from attire_compliance.detection.prelabel import _draw_preview
from attire_compliance.paths import resolve_project_path

_ROBOFLOW_SUFFIX_RE = re.compile(
    r"^(?P<stem>.+?)_(?P<ext>jpg|jpeg|png|webp)\.rf\.[A-Za-z0-9]+\.(?:jpg|jpeg|png|webp)$",
    re.IGNORECASE,
)


def strip_roboflow_filename(file_name: str, extra_name: str | None = None) -> str:
    """Kembalikan nama file asli frame lokal tanpa akhiran hash `.rf.<hash>` Roboflow."""
    if extra_name:
        return Path(extra_name).name
    base = Path(file_name).name
    match = _ROBOFLOW_SUFFIX_RE.match(base)
    if match:
        return f"{match.group('stem')}.{match.group('ext').lower()}"
    return base


def discover_coco_exports(interim_dir: Path) -> list[Path]:
    """Temukan seluruh folder ekspor COCO (mengandung `*.coco.json`) di `data/interim/`."""
    found: list[Path] = []
    if not interim_dir.is_dir():
        return found
    for child in sorted(interim_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in ("frames", "annotations", "roboflow_upload"):
            continue
        if list(child.rglob("*.coco.json")):
            found.append(child)
    return found


def _find_coco_json_files(source_path: Path) -> list[Path]:
    if source_path.is_file():
        return [source_path]
    if source_path.is_dir():
        jsons = sorted(source_path.rglob("*.coco.json"))
        if jsons:
            return jsons
        return sorted(source_path.rglob("*.json"))
    raise FileNotFoundError(f"Sumber anotasi COCO tidak ditemukan: {source_path}")


def _coco_bbox_to_yolo(
    bbox: list[float] | tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    """Konversi `[x, y, w, h]` piksel COCO menjadi `(xc, yc, bw, bh)` ternormalisasi `[0, 1]`."""
    if len(bbox) != 4 or img_w <= 0 or img_h <= 0:
        return None
    x, y, w, h = (float(v) for v in bbox)
    x1 = max(0.0, min(float(img_w), x))
    y1 = max(0.0, min(float(img_h), y))
    x2 = max(0.0, min(float(img_w), x + w))
    y2 = max(0.0, min(float(img_h), y + h))
    bw = (x2 - x1) / float(img_w)
    bh = (y2 - y1) / float(img_h)
    if bw <= 1e-6 or bh <= 1e-6:
        return None
    xc = ((x1 + x2) * 0.5) / float(img_w)
    yc = ((y1 + y2) * 0.5) / float(img_h)
    return (
        max(0.0, min(1.0, xc)),
        max(0.0, min(1.0, yc)),
        max(1e-6, min(1.0, bw)),
        max(1e-6, min(1.0, bh)),
    )


def import_local_coco_annotations(
    *,
    sources: list[Path] | None = None,
    frames_dir: Path | None = None,
    annotations_dir: Path | None = None,
    max_previews: int = 8,
    cleanup_interim: bool = False,
    configs_dir: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Gabungkan 1+ folder ekspor COCO Roboflow menjadi label YOLO lokal bersih.

    Output tersimpan di `data/interim/annotations/`:
    - `<batch>/<stem>.txt`: label YOLO 6-kelas terverifikasi sejajar `data/interim/frames/`
    - `classes.txt`: daftar 6 kelas terkunci sesuai `configs/classes.yaml`
    - `source_coco/merged_local_annotations.coco.json`: arsip COCO gabungan bebas duplikat
    - `verified_previews/*.jpg`: sampel overlay visual untuk sanity check koordinat
    - `import_report.json`: statistik lengkap penggabungan & validasi
    """
    classes_cfg = load_classes_config(configs_dir)
    name_to_id = {name.lower(): cid for name, cid in classes_cfg.name_to_id.items()}
    id_to_name = classes_cfg.id_to_name

    interim_dir = resolve_project_path("data/interim", project_root)
    resolved_frames = (
        frames_dir.resolve()
        if frames_dir is not None
        else resolve_project_path("data/interim/frames", project_root)
    )
    resolved_ann = (
        annotations_dir.resolve()
        if annotations_dir is not None
        else resolve_project_path("data/interim/annotations", project_root)
    )

    manifest_path = resolved_frames / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest frame lokal tidak ditemukan: {manifest_path}. "
            "Jalankan `attire data extract-frames` lebih dulu."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_lookup: dict[str, dict[str, Any]] = {}
    stem_lookup: dict[str, str] = {}
    for rec in manifest.get("frames", []):
        rel_file = str(rec["file"]).replace("\\", "/")
        img_path = resolved_frames / rel_file
        if not img_path.is_file():
            continue
        with Image.open(img_path) as im:
            disk_w, disk_h = im.size
        batch = Path(rel_file).parent.as_posix()
        frame_lookup[img_path.name] = {
            "rel_file": rel_file,
            "batch": batch,
            "group_id": str(rec.get("group_id") or ""),
            "path": img_path,
            "width": disk_w,
            "height": disk_h,
        }
        stem_lookup[img_path.stem.lower()] = img_path.name

    source_paths = (
        [p.resolve() for p in sources] if sources else discover_coco_exports(interim_dir)
    )
    if not source_paths:
        raise FileNotFoundError(
            f"Tidak ditemukan folder ekspor COCO di {interim_dir}. "
            "Berikan path eksplisit melalui opsi `--source`."
        )

    # Simpan entri terbaik per frame lokal: rel_file -> record gabungan
    merged_records: dict[str, dict[str, Any]] = {}
    source_stats: list[dict[str, Any]] = []
    unmatched_images: list[str] = []
    unmapped_categories: Counter[str] = Counter()
    dimension_mismatches: list[dict[str, Any]] = []
    invalid_boxes = 0
    overlaps: list[dict[str, Any]] = []

    for src_path in source_paths:
        json_files = _find_coco_json_files(src_path)
        src_img_count = 0
        src_ann_count = 0

        for jf in json_files:
            raw = json.loads(jf.read_text(encoding="utf-8"))
            cat_id_to_target: dict[int, int] = {}
            cat_id_to_raw_name: dict[int, str] = {}
            for cat in raw.get("categories", []):
                cid = int(cat["id"])
                cname = str(cat.get("name", "")).strip()
                cat_id_to_raw_name[cid] = cname
                mapped = name_to_id.get(cname.lower())
                if mapped is not None:
                    cat_id_to_target[cid] = mapped

            anns_by_image: dict[int, list[dict[str, Any]]] = {}
            for ann in raw.get("annotations", []):
                anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

            for img_entry in raw.get("images", []):
                src_img_count += 1
                extra = img_entry.get("extra") or {}
                orig_name = strip_roboflow_filename(
                    str(img_entry.get("file_name", "")),
                    extra.get("name") if isinstance(extra, dict) else None,
                )

                frame_info = frame_lookup.get(orig_name)
                if frame_info is None:
                    matched_key = stem_lookup.get(Path(orig_name).stem.lower())
                    if matched_key is not None:
                        frame_info = frame_lookup[matched_key]

                if frame_info is None:
                    unmatched_images.append(orig_name)
                    continue

                coco_w = int(img_entry.get("width", frame_info["width"]))
                coco_h = int(img_entry.get("height", frame_info["height"]))
                if (coco_w, coco_h) != (frame_info["width"], frame_info["height"]):
                    dimension_mismatches.append(
                        {
                            "file": frame_info["rel_file"],
                            "coco": [coco_w, coco_h],
                            "disk": [frame_info["width"], frame_info["height"]],
                        }
                    )

                yolo_boxes: list[tuple[int, float, float, float, float]] = []
                coco_boxes: list[dict[str, Any]] = []
                for ann in anns_by_image.get(int(img_entry["id"]), []):
                    raw_cid = int(ann["category_id"])
                    target_cid = cat_id_to_target.get(raw_cid)
                    if target_cid is None:
                        unmapped_categories[cat_id_to_raw_name.get(raw_cid, str(raw_cid))] += 1
                        continue
                    yolo_box = _coco_bbox_to_yolo(ann.get("bbox", []), coco_w, coco_h)
                    if yolo_box is None:
                        invalid_boxes += 1
                        continue
                    xc, yc, bw, bh = yolo_box
                    yolo_boxes.append((target_cid, xc, yc, bw, bh))
                    # Simpan bbox bersih dalam skala dimensi disk aktual
                    px_w = bw * frame_info["width"]
                    px_h = bh * frame_info["height"]
                    px_x = (xc - bw * 0.5) * frame_info["width"]
                    px_y = (yc - bh * 0.5) * frame_info["height"]
                    coco_boxes.append(
                        {
                            "category_id": target_cid,
                            "bbox": [
                                round(px_x, 2),
                                round(px_y, 2),
                                round(px_w, 2),
                                round(px_h, 2),
                            ],
                            "area": round(px_w * px_h, 2),
                            "iscrowd": 0,
                        }
                    )
                    src_ann_count += 1

                rel_file = frame_info["rel_file"]
                new_rec = {
                    "frame_info": frame_info,
                    "yolo_boxes": yolo_boxes,
                    "coco_boxes": coco_boxes,
                    "source": src_path.name,
                }
                if rel_file in merged_records:
                    prev = merged_records[rel_file]
                    keep_new = len(yolo_boxes) >= len(prev["yolo_boxes"])
                    overlaps.append(
                        {
                            "file": rel_file,
                            "prev_source": prev["source"],
                            "prev_boxes": len(prev["yolo_boxes"]),
                            "new_source": src_path.name,
                            "new_boxes": len(yolo_boxes),
                            "kept": src_path.name if keep_new else prev["source"],
                        }
                    )
                    if keep_new:
                        merged_records[rel_file] = new_rec
                else:
                    merged_records[rel_file] = new_rec

        source_stats.append(
            {
                "source": src_path.name,
                "json_files": len(json_files),
                "images": src_img_count,
                "annotations": src_ann_count,
            }
        )

    if not merged_records:
        raise RuntimeError("Tidak ada anotasi COCO yang cocok dengan frame lokal di manifest.json.")

    # Bersihkan folder batch lama di annotations_dir agar tidak tercampur label usang
    resolved_ann.mkdir(parents=True, exist_ok=True)
    for batch_name in sorted({info["batch"] for info in frame_lookup.values()}):
        batch_dir = resolved_ann / batch_name
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        batch_dir.mkdir(parents=True, exist_ok=True)

    # Tulis label YOLO + susun arsip COCO gabungan yang bersih (ID kelas 0..5 terkunci)
    merged_coco: dict[str, Any] = {
        "info": {
            "description": "Merged & validated Polibatam local attire compliance annotations",
            "version": "1.0",
        },
        "categories": [
            {"id": cid, "name": cname, "supercategory": "attire"}
            for cid, cname in sorted(id_to_name.items())
        ],
        "images": [],
        "annotations": [],
    }

    total_instances: Counter[str] = Counter()
    by_batch_images: Counter[str] = Counter()
    by_batch_instances: dict[str, Counter[str]] = {}
    empty_annotated: list[str] = []
    ann_id_seq = 1

    sorted_rel_files = sorted(merged_records.keys())
    for img_idx, rel_file in enumerate(sorted_rel_files, start=1):
        rec = merged_records[rel_file]
        finfo = rec["frame_info"]
        yolo_boxes = rec["yolo_boxes"]
        coco_boxes = rec["coco_boxes"]

        if not yolo_boxes:
            empty_annotated.append(rel_file)

        txt_path = (resolved_ann / rel_file).with_suffix(".txt")
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            for cid, xc, yc, bw, bh in yolo_boxes
        ]
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        batch = finfo["batch"]
        by_batch_images[batch] += 1
        by_batch_instances.setdefault(batch, Counter())

        merged_coco["images"].append(
            {
                "id": img_idx,
                "file_name": rel_file,
                "width": finfo["width"],
                "height": finfo["height"],
                "group_id": finfo["group_id"],
            }
        )
        for box in coco_boxes:
            cls_name = id_to_name[int(box["category_id"])]
            total_instances[cls_name] += 1
            by_batch_instances[batch][cls_name] += 1
            merged_coco["annotations"].append(
                {
                    "id": ann_id_seq,
                    "image_id": img_idx,
                    **box,
                }
            )
            ann_id_seq += 1

    # Simpan arsip COCO gabungan & classes.txt
    coco_archive_dir = resolved_ann / "source_coco"
    coco_archive_dir.mkdir(parents=True, exist_ok=True)
    merged_coco_path = coco_archive_dir / "merged_local_annotations.coco.json"
    merged_coco_path.write_text(
        json.dumps(merged_coco, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (resolved_ann / "classes.txt").write_text(
        "\n".join(classes_cfg.names) + "\n", encoding="utf-8"
    )

    # Buat sampel preview visual overlay untuk verifikasi koordinat
    preview_dir = resolved_ann / "verified_previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_files: list[str] = []
    if max_previews > 0 and sorted_rel_files:
        preview_dir.mkdir(parents=True, exist_ok=True)
        # Prioritaskan perwakilan tiap batch + foto sandal + foto hasil rotasi
        priority_keys = [
            k
            for k in sorted_rel_files
            if any(cid == 5 for cid, *_ in merged_records[k]["yolo_boxes"])
            or k.endswith(("almet_img_6044.jpg", "nonalmet_img_0501.jpg"))
        ]
        step = max(1, len(sorted_rel_files) // max_previews)
        sampled = list(dict.fromkeys(priority_keys + sorted_rel_files[::step]))[:max_previews]
        for rel_file in sampled:
            rec = merged_records[rel_file]
            finfo = rec["frame_info"]
            img_bgr = cv2.imread(str(finfo["path"]))
            if img_bgr is None:
                continue
            ih, iw = img_bgr.shape[:2]
            dets = [
                Detection(
                    class_id=cid,
                    class_name=id_to_name[cid],
                    confidence=1.0,
                    bbox=(
                        (xc - bw * 0.5) * iw,
                        (yc - bh * 0.5) * ih,
                        (xc + bw * 0.5) * iw,
                        (yc + bh * 0.5) * ih,
                    ),
                )
                for cid, xc, yc, bw, bh in rec["yolo_boxes"]
            ]
            rendered = _draw_preview(img_bgr, dets)
            out_name = rel_file.replace("/", "_")
            cv2.imwrite(str(preview_dir / out_name), rendered)
            preview_files.append(out_name)

    missing_frames = sorted(
        info["rel_file"]
        for info in frame_lookup.values()
        if info["rel_file"] not in merged_records
    )

    cleaned_paths: list[str] = []
    if cleanup_interim:
        # Hapus folder ekspor mentah Roboflow, bundle upload sementara, dan prelabel v0 lama
        # (aman karena arsip COCO gabungan bersih sudah tersimpan di source_coco/)
        for src_path in source_paths:
            if src_path.exists() and interim_dir.resolve() in src_path.resolve().parents:
                if src_path.is_dir():
                    shutil.rmtree(src_path)
                else:
                    src_path.unlink()
                cleaned_paths.append(src_path.name)

        for stale in (
            interim_dir / "roboflow_upload",
            resolved_ann / "prelabels",
            resolved_ann / "previews",
            resolved_ann / "prelabel_report.json",
        ):
            if stale.is_dir():
                shutil.rmtree(stale)
                cleaned_paths.append(str(stale.relative_to(interim_dir)))
            elif stale.is_file():
                stale.unlink()
                cleaned_paths.append(str(stale.relative_to(interim_dir)))

    report: dict[str, Any] = {
        "sources": source_stats,
        "total_manifest_frames": len(frame_lookup),
        "annotated_images": len(merged_records),
        "missing_frames": missing_frames,
        "empty_annotated_images": empty_annotated,
        "total_annotations": int(sum(total_instances.values())),
        "instances": {name: int(total_instances.get(name, 0)) for name in classes_cfg.names},
        "by_batch": {
            batch: {
                "images": int(by_batch_images[batch]),
                "instances": {
                    name: int(by_batch_instances.get(batch, Counter()).get(name, 0))
                    for name in classes_cfg.names
                },
            }
            for batch in sorted(by_batch_images)
        },
        "overlaps": overlaps,
        "unmatched_images": unmatched_images,
        "unmapped_categories": dict(unmapped_categories),
        "dimension_mismatches": dimension_mismatches,
        "invalid_boxes_skipped": invalid_boxes,
        "annotations_dir": str(resolved_ann),
        "merged_coco_path": str(merged_coco_path),
        "preview_files": preview_files,
        "cleaned_paths": cleaned_paths,
    }
    (resolved_ann / "import_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
