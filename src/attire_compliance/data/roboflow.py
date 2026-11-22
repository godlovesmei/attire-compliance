"""Validasi, unpack, dan remap export dataset Roboflow Universe ke 6 kelas terkunci."""

from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from attire_compliance.config import load_all_configs
from attire_compliance.paths import resolve_project_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def _extract_roboflow_zips(roboflow_root: Path) -> list[str]:
    """Ekstrak file .zip yang diletakkan langsung di data/raw/roboflow/."""
    extracted: list[str] = []
    for zip_path in sorted(roboflow_root.glob("*.zip")):
        target_dir = roboflow_root / zip_path.stem
        if target_dir.exists() and any(target_dir.iterdir()):
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        extracted.append(zip_path.stem)
    return extracted


def _load_source_class_names(data_yaml_path: Path) -> dict[int, str]:
    with data_yaml_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    names = raw.get("names")
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise ValueError(f"Field `names` tidak ditemukan atau tidak valid di {data_yaml_path}")


def _find_split_dirs(dataset_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Deteksi layout Roboflow (`train/images` atau `images/train`)."""
    found: dict[str, tuple[Path, Path]] = {}
    for src_split, norm_split in SPLIT_MAP.items():
        cand_img_1 = dataset_dir / src_split / "images"
        cand_lbl_1 = dataset_dir / src_split / "labels"
        if cand_img_1.is_dir() and cand_lbl_1.is_dir():
            found[norm_split] = (cand_img_1, cand_lbl_1)
            continue
        cand_img_2 = dataset_dir / "images" / src_split
        cand_lbl_2 = dataset_dir / "labels" / src_split
        if cand_img_2.is_dir() and cand_lbl_2.is_dir():
            found[norm_split] = (cand_img_2, cand_lbl_2)
    return found


def import_roboflow_datasets(
    *,
    slug: str | None = None,
    configs_dir: Path | None = None,
    roboflow_root_override: Path | None = None,
) -> dict[str, Any]:
    """Remap dataset Roboflow di `data/raw/roboflow/<slug>/` ke format YOLO 6-kelas terstandar.

    Hasil remap disimpan di `data/raw/roboflow/<slug>/_remapped/` agar file asli tetap utuh.
    """
    cfgs = load_all_configs(configs_dir)
    rf_cfg = cfgs.sources.roboflow
    classes_cfg = cfgs.classes
    name_to_id = classes_cfg.name_to_id

    roboflow_root = (
        roboflow_root_override
        if roboflow_root_override is not None
        else resolve_project_path(rf_cfg.root_dir)
    )
    roboflow_root.mkdir(parents=True, exist_ok=True)
    extracted_zips = _extract_roboflow_zips(roboflow_root)

    if slug is not None:
        candidate_dirs = [roboflow_root / slug]
    else:
        candidate_dirs = sorted(
            d for d in roboflow_root.iterdir() if d.is_dir() and not d.name.startswith("_")
        )

    report: dict[str, Any] = {
        "source": "roboflow",
        "extracted_zips": extracted_zips,
        "datasets": {},
        "total_images_saved": 0,
        "total_instances": {name: 0 for name in classes_cfg.names},
    }

    for ds_dir in candidate_dirs:
        if not ds_dir.is_dir():
            raise FileNotFoundError(f"Direktori dataset Roboflow tidak ditemukan: {ds_dir}")

        ds_slug = ds_dir.name
        data_yaml_path = ds_dir / "data.yaml"
        if not data_yaml_path.is_file():
            report["datasets"][ds_slug] = {
                "status": "skipped",
                "reason": "data.yaml tidak ditemukan",
            }
            continue

        cfg_entry = rf_cfg.datasets.get(ds_slug)
        if cfg_entry is None:
            report["datasets"][ds_slug] = {
                "status": "skipped",
                "reason": (
                    f"Slug `{ds_slug}` belum terdaftar di configs/sources.yaml -> roboflow.datasets"
                ),
            }
            continue

        lower_class_map = {k.strip().lower(): v for k, v in cfg_entry.class_map.items()}
        src_id_to_name = _load_source_class_names(data_yaml_path)
        split_dirs = _find_split_dirs(ds_dir)

        out_dir = ds_dir / "_remapped"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ds_saved = 0
        ds_skipped_empty = 0
        ds_counts: Counter[str] = Counter()
        unmapped_classes: Counter[str] = Counter()
        invalid_lines = 0

        for norm_split, (src_img_dir, src_lbl_dir) in split_dirs.items():
            dst_img_dir = out_dir / "images" / norm_split
            dst_lbl_dir = out_dir / "labels" / norm_split
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(src_img_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.is_file():
                    ds_skipped_empty += 1
                    continue

                remapped_lines: list[str] = []
                for raw_line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = raw_line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        invalid_lines += 1
                        continue
                    try:
                        src_cls_id = int(float(parts[0]))
                        coords = [float(v) for v in parts[1:5]]
                    except ValueError:
                        invalid_lines += 1
                        continue
                    if (
                        any(v < 0.0 or v > 1.0 for v in coords)
                        or coords[2] <= 0.0
                        or coords[3] <= 0.0
                    ):
                        invalid_lines += 1
                        continue

                    src_cls_name = src_id_to_name.get(src_cls_id, f"id_{src_cls_id}")
                    target_name = lower_class_map.get(src_cls_name.strip().lower())
                    if target_name is None:
                        unmapped_classes[src_cls_name] += 1
                        continue

                    target_id = name_to_id[target_name]
                    xc, yc, bw, bh = coords
                    remapped_lines.append(
                        f"{target_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                    )
                    ds_counts[target_name] += 1

                if not remapped_lines:
                    ds_skipped_empty += 1
                    continue

                stem = f"rf_{ds_slug}_{norm_split}_{img_path.stem}"
                shutil.copy2(img_path, dst_img_dir / f"{stem}{img_path.suffix.lower()}")
                (dst_lbl_dir / f"{stem}.txt").write_text(
                    "\n".join(remapped_lines) + "\n", encoding="utf-8"
                )
                ds_saved += 1

        ds_class_dict = {name: int(ds_counts.get(name, 0)) for name in classes_cfg.names}
        yaml_payload = {
            "path": str(out_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": classes_cfg.id_to_name,
        }
        with (out_dir / "data.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(yaml_payload, handle, sort_keys=False, allow_unicode=True)

        report["datasets"][ds_slug] = {
            "status": "ok",
            "saved": ds_saved,
            "skipped_empty": ds_skipped_empty,
            "invalid_lines": invalid_lines,
            "instances": ds_class_dict,
            "unmapped_classes": dict(sorted(unmapped_classes.items())),
        }
        report["total_images_saved"] += ds_saved
        for name, count in ds_class_dict.items():
            report["total_instances"][name] += count

    (roboflow_root / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
