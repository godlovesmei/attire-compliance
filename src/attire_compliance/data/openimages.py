"""Auto-fetch kelas pelengkap (khususnya `Sandal`) dari Open Images V7 via FiftyOne."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from attire_compliance.config import ClassesConfig, load_all_configs
from attire_compliance.data.geometry import clip_bbox, xyxy_to_yolo
from attire_compliance.paths import resolve_project_path

SPLIT_ALIAS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def _write_yolo_data_yaml(output_dir: Path, classes_cfg: ClassesConfig) -> None:
    payload = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": classes_cfg.id_to_name,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def fetch_openimages(
    *,
    classes: list[str] | None = None,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
    configs_dir: Path | None = None,
    output_dir_override: Path | None = None,
) -> dict[str, Any]:
    """Tarik subset Open Images V7 via FiftyOne lalu ekspor ke format YOLO 6-kelas.

    Secara default hanya menarik kelas `Sandal` untuk melengkapi Fashionpedia (dan mencegah
    gambar `Trousers`/`Jeans` Open Images yang anotasi alas kakinya parsial masuk tanpa label).
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("fiftyone"):
            raise RuntimeError(
                "Library `fiftyone` belum terpasang. Jalankan: uv sync --extra data"
            ) from exc
        raise

    cfgs = load_all_configs(configs_dir)
    oi_cfg = cfgs.sources.openimages
    classes_cfg = cfgs.classes
    output_dir = (
        output_dir_override
        if output_dir_override is not None
        else resolve_project_path(oi_cfg.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simpan cache FiftyOne Zoo di dalam direktori raw proyek
    zoo_cache_dir = output_dir / "_zoo_cache"
    zoo_cache_dir.mkdir(parents=True, exist_ok=True)
    fo.config.dataset_zoo_dir = str(zoo_cache_dir.resolve())

    selected_classes = classes if classes else ["Sandal"]
    for oi_cls in selected_classes:
        if oi_cls not in oi_cfg.class_map:
            raise ValueError(
                f"Kelas Open Images `{oi_cls}` tidak ada di configs/sources.yaml. "
                f"Pilihan valid: {list(oi_cfg.class_map.keys())}"
            )
    selected_set = set(selected_classes)

    split_limits = {
        "train": max_train
        if max_train is not None
        else oi_cfg.max_samples_per_split.get("train", 600),
        "validation": max_val
        if max_val is not None
        else oi_cfg.max_samples_per_split.get("validation", 150),
        "test": max_test
        if max_test is not None
        else oi_cfg.max_samples_per_split.get("test", 150),
    }

    name_to_id = classes_cfg.name_to_id
    report: dict[str, Any] = {
        "source": "openimages",
        "zoo_name": oi_cfg.zoo_name,
        "requested_classes": selected_classes,
        "partial_annotation_warning": (
            "Open Images V7 tidak dianotasi secara exhaustif untuk semua 6 kelas. "
            "Direkomendasikan hanya mengambil `Sandal` sebagai pelengkap, lalu jalankan "
            "pseudo-labeling + review sebelum menggabungkan ke dataset v1."
        ),
        "splits": {},
        "total_images_saved": 0,
        "total_instances": {name: 0 for name in classes_cfg.names},
    }

    for oi_split, max_samples in split_limits.items():
        local_split = SPLIT_ALIAS[oi_split]
        img_dir = output_dir / "images" / local_split
        lbl_dir = output_dir / "labels" / local_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        if max_samples <= 0:
            report["splits"][local_split] = {
                "saved": 0,
                "skipped_empty": 0,
                "instances": {name: 0 for name in classes_cfg.names},
            }
            continue

        ds_name = f"attire_oi_{oi_split}_{'_'.join(selected_classes).lower()}"
        if ds_name in fo.list_datasets():
            fo.delete_dataset(ds_name)

        dataset = foz.load_zoo_dataset(
            oi_cfg.zoo_name,
            split=oi_split,
            label_types=["detections"],
            classes=selected_classes,
            max_samples=max_samples,
            only_matching=True,
            dataset_name=ds_name,
            shuffle=True,
            seed=42,
        )

        saved = 0
        skipped_empty = 0
        class_counts: Counter[str] = Counter()

        try:
            for sample in dataset:
                src_path = Path(sample.filepath)
                if not src_path.is_file():
                    continue
                detections_field = sample.ground_truth
                if detections_field is None or not detections_field.detections:
                    skipped_empty += 1
                    continue

                with Image.open(src_path) as im:
                    img_w, img_h = im.size

                yolo_lines: list[str] = []
                for det in detections_field.detections:
                    if det.label not in selected_set:
                        continue
                    target_name = oi_cfg.class_map.get(det.label)
                    if target_name is None:
                        continue
                    # FiftyOne bbox format: [x_top_left, y_top_left, width, height] normalized 0..1
                    rx, ry, rw, rh = det.bounding_box
                    clipped = clip_bbox(
                        (rx * img_w, ry * img_h, (rx + rw) * img_w, (ry + rh) * img_h),
                        img_w,
                        img_h,
                    )
                    if clipped is None:
                        continue
                    xc, yc, w, h = xyxy_to_yolo(clipped, img_w, img_h)
                    cls_id = name_to_id[target_name]
                    yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    class_counts[target_name] += 1

                if not yolo_lines:
                    skipped_empty += 1
                    continue

                stem = f"oi_{local_split}_{src_path.stem}"
                dst_img = img_dir / f"{stem}{src_path.suffix.lower() or '.jpg'}"
                dst_lbl = lbl_dir / f"{stem}.txt"
                shutil.copy2(src_path, dst_img)
                dst_lbl.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
                saved += 1
        finally:
            if ds_name in fo.list_datasets():
                fo.delete_dataset(ds_name)

        split_class_dict = {name: int(class_counts.get(name, 0)) for name in classes_cfg.names}
        report["splits"][local_split] = {
            "saved": saved,
            "skipped_empty": skipped_empty,
            "instances": split_class_dict,
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
