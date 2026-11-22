"""Auto-fetch dan pemetaan dataset Fashionpedia (Hugging Face) ke 6 kelas terkunci."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from attire_compliance.config import (
    ClassesConfig,
    FashionpediaRule,
    FashionpediaSourceConfig,
    load_all_configs,
)
from attire_compliance.data.geometry import (
    BBox,
    clip_bbox,
    expand_box_top,
    ioa,
    is_long_skirt,
    xyxy_to_yolo,
)
from attire_compliance.paths import resolve_project_path

# Kategori pakaian utama (bukan part kecil) untuk menghitung outfit vertical span
MAIN_GARMENT_CATEGORIES: frozenset[str] = frozenset(
    {
        "shirt, blouse",
        "top, t-shirt, sweatshirt",
        "sweater",
        "cardigan",
        "jacket",
        "vest",
        "pants",
        "shorts",
        "skirt",
        "coat",
        "dress",
        "jumpsuit",
        "shoe",
    }
)


@dataclass(frozen=True)
class RawAnnotation:
    category: str
    bbox: BBox


@dataclass(frozen=True)
class MappedAnnotation:
    class_id: int
    class_name: str
    bbox: BBox
    source_category: str


@dataclass(frozen=True)
class ImageMappingResult:
    annotations: list[MappedAnnotation]
    raw_categories: list[str]
    rejected_reasons: Counter[str]


def _has_part_overlap(
    garment_box: BBox,
    part_boxes: list[BBox],
    min_ioa: float,
    *,
    is_exclude: bool = False,
) -> bool:
    if not part_boxes:
        return False
    target_box = expand_box_top(garment_box) if is_exclude else garment_box
    threshold = (min_ioa * 0.5) if is_exclude else min_ioa
    return any(ioa(pbox, target_box) >= threshold for pbox in part_boxes)


def map_fashionpedia_objects(
    raw_objects: list[RawAnnotation],
    img_w: int,
    img_h: int,
    fp_cfg: FashionpediaSourceConfig,
    classes_cfg: ClassesConfig,
) -> ImageMappingResult:
    """Petakan anotasi mentah Fashionpedia satu gambar ke 6 kelas proyek."""
    cleaned: list[RawAnnotation] = []
    for obj in raw_objects:
        clipped = clip_bbox(obj.bbox, img_w, img_h)
        if clipped is not None:
            cleaned.append(RawAnnotation(category=obj.category, bbox=clipped))

    parts_by_name: dict[str, list[BBox]] = {}
    main_garment_boxes: list[BBox] = []
    raw_categories: list[str] = []

    for obj in cleaned:
        raw_categories.append(obj.category)
        parts_by_name.setdefault(obj.category, []).append(obj.bbox)
        if obj.category in MAIN_GARMENT_CATEGORIES:
            main_garment_boxes.append(obj.bbox)

    rules_by_source: dict[str, list[FashionpediaRule]] = {}
    for rule in fp_cfg.rules:
        rules_by_source.setdefault(rule.source_category, []).append(rule)

    name_to_id = classes_cfg.name_to_id
    mapped: list[MappedAnnotation] = []
    rejected: Counter[str] = Counter()

    for obj in cleaned:
        candidate_rules = rules_by_source.get(obj.category)
        if not candidate_rules:
            continue

        matched_rule: FashionpediaRule | None = None
        last_reject_reason: str | None = None

        for rule in candidate_rules:
            # Cek exclude_parts terlebih dahulu agar penolakan `hood` tercatat jelas
            hit_excl = [
                part
                for part in rule.exclude_parts
                if _has_part_overlap(
                    obj.bbox,
                    parts_by_name.get(part, []),
                    fp_cfg.part_overlap_min_ioa,
                    is_exclude=True,
                )
            ]
            if hit_excl:
                last_reject_reason = f"{obj.category}:excluded_by_{'+'.join(hit_excl)}"
                continue

            missing_req = [
                part
                for part in rule.require_parts
                if not _has_part_overlap(
                    obj.bbox,
                    parts_by_name.get(part, []),
                    fp_cfg.part_overlap_min_ioa,
                    is_exclude=False,
                )
            ]
            if missing_req:
                last_reject_reason = f"{obj.category}:missing_{'+'.join(missing_req)}"
                continue

            if rule.long_only and not is_long_skirt(
                obj.bbox,
                img_h=img_h,
                all_garment_boxes=main_garment_boxes,
                min_img_height_ratio=fp_cfg.long_skirt_min_height_ratio,
            ):
                last_reject_reason = f"{obj.category}:too_short_for_long_skirt"
                continue

            matched_rule = rule
            break

        if matched_rule is not None:
            cls_name = matched_rule.target_class
            mapped.append(
                MappedAnnotation(
                    class_id=name_to_id[cls_name],
                    class_name=cls_name,
                    bbox=obj.bbox,
                    source_category=obj.category,
                )
            )
        elif last_reject_reason is not None:
            rejected[last_reject_reason] += 1

    return ImageMappingResult(
        annotations=mapped,
        raw_categories=sorted(set(raw_categories)),
        rejected_reasons=rejected,
    )


def _extract_category_names(dataset_obj: Any) -> list[str]:
    """Ambil daftar nama kategori resmi dari fitur dataset Hugging Face."""
    features = getattr(dataset_obj, "features", None)
    if features is None:
        raise RuntimeError("Dataset Hugging Face tidak memiliki metadata `features`.")
    cat_feat = features["objects"]["category"]
    inner = getattr(cat_feat, "feature", cat_feat)
    names = getattr(inner, "names", None)
    if not names:
        raise RuntimeError("Tidak menemukan daftar nama kategori pada Fashionpedia features.")
    return list(names)


def _parse_hf_record(
    record: dict[str, Any],
    category_names: list[str],
    img_w: int,
    img_h: int,
) -> list[RawAnnotation]:
    """Parse bbox HF Fashionpedia serta rescale jika dimensi record != ukuran image aktual."""
    rec_w = record.get("width")
    rec_h = record.get("height")
    scale_x = (img_w / rec_w) if isinstance(rec_w, (int, float)) and rec_w > 0 else 1.0
    scale_y = (img_h / rec_h) if isinstance(rec_h, (int, float)) and rec_h > 0 else 1.0

    objects = record["objects"]
    bboxes = objects["bbox"]
    categories = objects["category"]
    parsed: list[RawAnnotation] = []
    for bbox, cat_id in zip(bboxes, categories, strict=True):
        idx = int(cat_id)
        if not (0 <= idx < len(category_names)):
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        parsed.append(
            RawAnnotation(
                category=category_names[idx],
                bbox=(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
            )
        )
    return parsed


def _write_yolo_data_yaml(output_dir: Path, classes_cfg: ClassesConfig) -> None:
    payload = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": classes_cfg.id_to_name,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def fetch_fashionpedia(
    *,
    max_train: int = 2000,
    max_val: int = 500,
    streaming: bool = True,
    configs_dir: Path | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Tarik subset Fashionpedia dari Hugging Face, filter part, dan ekspor ke format YOLO."""
    try:
        from datasets import get_dataset_split_names, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Library `datasets` belum terpasang. Jalankan: uv sync --extra data"
        ) from exc

    cfgs = load_all_configs(configs_dir)
    fp_cfg = cfgs.sources.fashionpedia
    classes_cfg = cfgs.classes
    output_dir = (
        output_dir_override
        if output_dir_override is not None
        else resolve_project_path(fp_cfg.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    available_splits = set(get_dataset_split_names(fp_cfg.hf_dataset))
    val_hf_split = "val" if "val" in available_splits else "validation"
    if val_hf_split not in available_splits:
        raise RuntimeError(
            f"Split validasi tidak ditemukan di {fp_cfg.hf_dataset}: {sorted(available_splits)}"
        )

    split_specs = [
        ("train", "train", max_train),
        (val_hf_split, "val", max_val),
    ]

    report: dict[str, Any] = {
        "source": "fashionpedia",
        "hf_dataset": fp_cfg.hf_dataset,
        "streaming": streaming,
        "splits": {},
        "total_images_saved": 0,
        "total_instances": {name: 0 for name in classes_cfg.names},
    }

    for hf_split, local_split, limit in split_specs:
        img_dir = output_dir / "images" / local_split
        lbl_dir = output_dir / "labels" / local_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"{local_split}.jsonl"

        ds = load_dataset(fp_cfg.hf_dataset, split=hf_split, streaming=streaming)
        category_names = _extract_category_names(ds)

        scanned = 0
        saved = 0
        skipped_empty = 0
        class_counts: Counter[str] = Counter()
        source_breakdown: Counter[str] = Counter()
        rejected_totals: Counter[str] = Counter()

        with meta_path.open("w", encoding="utf-8") as meta_handle:
            for record in ds:
                if limit > 0 and saved >= limit:
                    break
                scanned += 1

                image: Image.Image = record["image"]
                if image.mode != "RGB":
                    image = image.convert("RGB")
                img_w, img_h = image.size
                image_id = str(record["image_id"])

                raw_objs = _parse_hf_record(record, category_names, img_w, img_h)
                mapped_res = map_fashionpedia_objects(raw_objs, img_w, img_h, fp_cfg, classes_cfg)
                rejected_totals.update(mapped_res.rejected_reasons)

                if not mapped_res.annotations:
                    skipped_empty += 1
                    continue

                stem = f"fp_{local_split}_{image_id}"
                img_path = img_dir / f"{stem}.jpg"
                lbl_path = lbl_dir / f"{stem}.txt"

                image.save(img_path, format="JPEG", quality=92)

                yolo_lines: list[str] = []
                for ann in mapped_res.annotations:
                    xc, yc, w, h = xyxy_to_yolo(ann.bbox, img_w, img_h)
                    yolo_lines.append(f"{ann.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    class_counts[ann.class_name] += 1
                    source_breakdown[f"{ann.source_category} -> {ann.class_name}"] += 1

                lbl_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

                meta_row = {
                    "image_id": stem,
                    "split": local_split,
                    "width": img_w,
                    "height": img_h,
                    "raw_categories": mapped_res.raw_categories,
                    "mapped_classes": sorted({a.class_name for a in mapped_res.annotations}),
                    "num_boxes": len(mapped_res.annotations),
                }
                meta_handle.write(json.dumps(meta_row, ensure_ascii=False) + "\n")
                saved += 1

        split_class_dict = {name: int(class_counts.get(name, 0)) for name in classes_cfg.names}
        report["splits"][local_split] = {
            "hf_split": hf_split,
            "scanned": scanned,
            "saved": saved,
            "skipped_empty": skipped_empty,
            "instances": split_class_dict,
            "source_breakdown": dict(sorted(source_breakdown.items())),
            "rejected_reasons": dict(sorted(rejected_totals.items())),
        }
        report["total_images_saved"] += saved
        for name, count in split_class_dict.items():
            report["total_instances"][name] += count

    _write_yolo_data_yaml(output_dir, classes_cfg)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
