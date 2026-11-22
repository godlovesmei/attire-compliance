"""Penggabungan, validasi label, deduplikasi, dan pembuatan dataset deteksi versi `vN`."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from attire_compliance.config import load_all_configs
from attire_compliance.data.split import (
    DatasetSample,
    assign_group_splits,
    deduplicate_samples,
    verify_no_group_leakage,
)
from attire_compliance.paths import resolve_project_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_yolo_label_file(
    label_path: Path,
    num_classes: int = 6,
) -> tuple[tuple[tuple[int, float, float, float, float], ...], int]:
    """Baca & validasi file label YOLO; kembalikan (daftar box valid, jumlah baris tidak valid)."""
    if not label_path.is_file():
        return (), 0
    valid: list[tuple[int, float, float, float, float]] = []
    invalid = 0
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            invalid += 1
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            invalid += 1
            continue
        if not (0 <= cls_id < num_classes):
            invalid += 1
            continue
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            invalid += 1
            continue
        valid.append((cls_id, xc, yc, bw, bh))
    return tuple(valid), invalid


def _collect_yolo_dir_samples(
    source_name: str,
    root_dir: Path,
    num_classes: int,
) -> tuple[list[DatasetSample], int]:
    """Kumpulkan pasangan gambar & label YOLO dari struktur `images/<split>` + `labels/<split>`."""
    samples: list[DatasetSample] = []
    invalid_lines = 0
    images_root = root_dir / "images"
    labels_root = root_dir / "labels"
    if not images_root.is_dir() or not labels_root.is_dir():
        return samples, invalid_lines

    for img_path in sorted(images_root.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img_path.relative_to(images_root)
        lbl_path = (labels_root / rel).with_suffix(".txt")
        labels, bad = parse_yolo_label_file(lbl_path, num_classes=num_classes)
        invalid_lines += bad
        if not labels:
            continue
        sample_id = f"{source_name}_{rel.parent.as_posix().replace('/', '_')}_{img_path.stem}"
        samples.append(
            DatasetSample(
                sample_id=sample_id,
                source=source_name,
                group_id=sample_id,  # tiap gambar publik adalah grup independen
                image_path=img_path,
                labels=labels,
            )
        )
    return samples, invalid_lines


def _collect_local_annotated_samples(
    frames_dir: Path,
    annotations_dir: Path,
    num_classes: int,
) -> tuple[list[DatasetSample], int]:
    """Kumpulkan frame lokal dari `data/interim/frames` yang sudah memiliki anotasi YOLO."""
    samples: list[DatasetSample] = []
    invalid_lines = 0
    manifest_path = frames_dir / "manifest.json"
    if not manifest_path.is_file() or not annotations_dir.is_dir():
        return samples, invalid_lines

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rec in manifest.get("frames", []):
        rel_file = rec["file"]  # mis. "almamater/almet_img_5969.jpg"
        img_path = frames_dir / rel_file
        if not img_path.is_file():
            continue

        rel_txt = Path(rel_file).with_suffix(".txt")
        # Cari anotasi baik terstruktur per batch maupun datar di annotations_dir
        candidates = [
            annotations_dir / rel_txt,
            annotations_dir / "labels" / rel_txt,
            annotations_dir / rel_txt.name,
            annotations_dir / "labels" / rel_txt.name,
        ]
        lbl_path = next((p for p in candidates if p.is_file()), None)
        if lbl_path is None:
            continue

        labels, bad = parse_yolo_label_file(lbl_path, num_classes=num_classes)
        invalid_lines += bad
        if not labels:
            continue

        stem_clean = rel_file.replace("/", "_").rsplit(".", 1)[0]
        samples.append(
            DatasetSample(
                sample_id=f"local_{stem_clean}",
                source="local",
                group_id=str(rec.get("group_id") or f"local_{stem_clean}"),
                image_path=img_path,
                labels=labels,
            )
        )
    return samples, invalid_lines


def _link_or_copy(src: Path, dst: Path, use_hardlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if use_hardlinks:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def build_detection_dataset(
    *,
    version: str = "v1",
    include_fashionpedia: bool = True,
    include_openimages: bool = True,
    include_roboflow: bool = True,
    include_local: bool = True,
    annotations_dir: Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    phash_threshold: int = 3,
    seed: int = 42,
    use_hardlinks: bool = True,
    configs_dir: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Bangun dataset deteksi versi `vN` di `data/processed/detection/<version>/`."""
    cfgs = load_all_configs(configs_dir)
    classes_cfg = cfgs.classes
    num_classes = len(classes_cfg.names)

    all_samples: list[DatasetSample] = []
    source_raw_counts: dict[str, int] = {}
    total_invalid_lines = 0

    # Prioritaskan data lokal lebih dulu agar tidak tersingkir saat dedup
    if include_local:
        frames_dir = resolve_project_path("data/interim/frames", project_root)
        ann_dir = (
            annotations_dir
            if annotations_dir is not None
            else resolve_project_path("data/interim/annotations", project_root)
        )
        local_samples, bad = _collect_local_annotated_samples(frames_dir, ann_dir, num_classes)
        source_raw_counts["local"] = len(local_samples)
        total_invalid_lines += bad
        all_samples.extend(local_samples)

    if include_fashionpedia:
        fp_dir = resolve_project_path(cfgs.sources.fashionpedia.output_dir, project_root)
        fp_samples, bad = _collect_yolo_dir_samples("fashionpedia", fp_dir, num_classes)
        source_raw_counts["fashionpedia"] = len(fp_samples)
        total_invalid_lines += bad
        all_samples.extend(fp_samples)

    if include_openimages:
        oi_dir = resolve_project_path(cfgs.sources.openimages.output_dir, project_root)
        oi_samples, bad = _collect_yolo_dir_samples("openimages", oi_dir, num_classes)
        source_raw_counts["openimages"] = len(oi_samples)
        total_invalid_lines += bad
        all_samples.extend(oi_samples)

    if include_roboflow:
        rf_root = resolve_project_path(cfgs.sources.roboflow.root_dir, project_root)
        rf_total = 0
        if rf_root.is_dir():
            for ds_dir in sorted(d for d in rf_root.iterdir() if d.is_dir()):
                remapped = ds_dir / "_remapped"
                if remapped.is_dir():
                    rf_samples, bad = _collect_yolo_dir_samples(
                        f"roboflow_{ds_dir.name}", remapped, num_classes
                    )
                    rf_total += len(rf_samples)
                    total_invalid_lines += bad
                    all_samples.extend(rf_samples)
        source_raw_counts["roboflow"] = rf_total

    if not all_samples:
        raise RuntimeError(
            "Tidak ada sampel berlabel yang ditemukan dari sumber yang diaktifkan."
        )

    deduped, duplicates = deduplicate_samples(all_samples, phash_threshold=phash_threshold)
    splits = assign_group_splits(
        deduped,
        num_classes=num_classes,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    leakage_count = verify_no_group_leakage(splits)
    if leakage_count != 0:
        raise RuntimeError(f"Terdeteksi group leakage antar split: {leakage_count} grup!")

    out_dir = resolve_project_path(f"data/processed/detection/{version}", project_root)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_name = classes_cfg.id_to_name
    split_summary: dict[str, dict[str, Any]] = {}
    total_instances = Counter[str]()
    fingerprint_hasher = hashlib.sha256()

    for split_name in ("train", "val", "test"):
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        cls_counts = Counter[str]()
        src_counts = Counter[str]()
        groups_in_split: set[tuple[str, str]] = set()

        for sample in sorted(splits[split_name], key=lambda s: s.sample_id):
            ext = sample.image_path.suffix.lower() or ".jpg"
            dst_img = img_dir / f"{sample.sample_id}{ext}"
            dst_lbl = lbl_dir / f"{sample.sample_id}.txt"
            _link_or_copy(sample.image_path, dst_img, use_hardlinks=use_hardlinks)

            lines = [
                f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                for cls_id, xc, yc, bw, bh in sample.labels
            ]
            dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

            src_counts[sample.source] += 1
            groups_in_split.add((sample.source, sample.group_id))
            fingerprint_hasher.update(
                f"{split_name}:{sample.sample_id}:{sample.group_id}\n".encode()
            )
            for cls_id, *_ in sample.labels:
                cls_name = id_to_name[cls_id]
                cls_counts[cls_name] += 1
                total_instances[cls_name] += 1

        split_summary[split_name] = {
            "images": len(splits[split_name]),
            "groups": len(groups_in_split),
            "by_source": dict(sorted(src_counts.items())),
            "instances": {name: int(cls_counts.get(name, 0)) for name in classes_cfg.names},
        }

    data_yaml_payload = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": id_to_name,
    }
    data_yaml_path = out_dir / "data.yaml"
    with data_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml_payload, handle, sort_keys=False, allow_unicode=True)

    manifest: dict[str, Any] = {
        "version": version,
        "output_dir": str(out_dir.resolve()),
        "data_yaml": str(data_yaml_path.resolve()),
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "phash_threshold": phash_threshold,
        "sources_collected": source_raw_counts,
        "total_collected": len(all_samples),
        "duplicates_removed": len(duplicates),
        "invalid_label_lines_skipped": total_invalid_lines,
        "total_images": len(deduped),
        "group_leakage_count": leakage_count,
        "splits": split_summary,
        "total_instances": {name: int(total_instances.get(name, 0)) for name in classes_cfg.names},
        "split_sha256": fingerprint_hasher.hexdigest(),
        "duplicates_sample": duplicates[:50],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
