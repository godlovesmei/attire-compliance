"""Unit test offline untuk Fase 4 (dedup, group split, build-detection) & Fase 5 (detektor)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from attire_compliance.config import load_all_configs
from attire_compliance.data.coco_import import (
    import_local_coco_annotations,
    strip_roboflow_filename,
)
from attire_compliance.data.merge import build_detection_dataset, parse_yolo_label_file
from attire_compliance.data.split import (
    DatasetSample,
    assign_group_splits,
    deduplicate_samples,
    verify_no_group_leakage,
)
from attire_compliance.detection.predictor import Detection
from attire_compliance.detection.prelabel import _draw_preview, nms_detections
from attire_compliance.detection.train import _validate_data_yaml


def _make_pattern_img(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = np.zeros((160, 160, 3), dtype=np.uint8)
    for _ in range(8):
        x0, y0 = int(rng.integers(0, 110)), int(rng.integers(0, 110))
        w, h = int(rng.integers(20, 50)), int(rng.integers(20, 50))
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        cv2.rectangle(arr, (x0, y0), (x0 + w, y0 + h), color, -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="JPEG", quality=92)


def test_parse_yolo_label_file_valid_and_invalid(tmp_path: Path) -> None:
    lbl = tmp_path / "sample.txt"
    lbl.write_text(
        "0 0.5 0.5 0.3 0.4\n"
        "5 0.2 0.8 0.1 0.1\n"
        "6 0.5 0.5 0.2 0.2\n"       # class_id out of range
        "1 1.2 0.5 0.2 0.2\n"       # coord > 1
        "2 0.5 0.5 0.0 0.2\n"       # width == 0
        "bad line\n",
        encoding="utf-8",
    )
    boxes, invalid = parse_yolo_label_file(lbl, num_classes=6)
    assert len(boxes) == 2
    assert boxes[0][0] == 0
    assert boxes[1][0] == 5
    assert invalid == 4


def test_deduplicate_and_group_split_zero_leakage(tmp_path: Path) -> None:
    samples: list[DatasetSample] = []
    # Buat 12 grup independen dengan beberapa sampel per grup
    for grp_idx in range(12):
        for frame_idx in range(3):
            seed = grp_idx * 10 + frame_idx + 1
            img_path = tmp_path / f"g{grp_idx}_f{frame_idx}.jpg"
            _make_pattern_img(img_path, seed)
            cls_id = grp_idx % 6
            samples.append(
                DatasetSample(
                    sample_id=f"local_g{grp_idx}_f{frame_idx}",
                    source="local",
                    group_id=f"local_g{grp_idx:02d}",
                    image_path=img_path,
                    labels=((cls_id, 0.5, 0.5, 0.3, 0.3),),
                )
            )

    # Tambahkan 1 duplikat persis dari sampel pertama
    dup_path = tmp_path / "dup.jpg"
    _make_pattern_img(dup_path, 1)
    samples.append(
        DatasetSample(
            sample_id="fp_dup",
            source="fashionpedia",
            group_id="fp_dup",
            image_path=dup_path,
            labels=((0, 0.5, 0.5, 0.3, 0.3),),
        )
    )

    deduped, dups = deduplicate_samples(samples, phash_threshold=3)
    assert len(dups) == 1
    assert dups[0]["dropped_id"] == "fp_dup"
    assert dups[0]["kept_id"] == "local_g0_f0"
    assert len(deduped) == 36

    splits = assign_group_splits(deduped, num_classes=6, seed=42)
    assert verify_no_group_leakage(splits) == 0
    assert sum(len(v) for v in splits.values()) == 36
    assert len(splits["train"]) > len(splits["val"]) > 0
    assert len(splits["test"]) > 0


def test_build_detection_dataset_end_to_end(tmp_path: Path) -> None:
    # 1. Fashionpedia sintetis (8 gambar)
    fp_img = tmp_path / "data/raw/fashionpedia/images/train"
    fp_lbl = tmp_path / "data/raw/fashionpedia/labels/train"
    fp_lbl.mkdir(parents=True)
    for i in range(8):
        _make_pattern_img(fp_img / f"fp_{i}.jpg", seed=100 + i)
        (fp_lbl / f"fp_{i}.txt").write_text(f"{i % 5} 0.5 0.5 0.3 0.4\n", encoding="utf-8")

    # 2. Open Images sintetis (4 gambar sandal)
    oi_img = tmp_path / "data/raw/openimages/images/train"
    oi_lbl = tmp_path / "data/raw/openimages/labels/train"
    oi_lbl.mkdir(parents=True)
    for i in range(4):
        _make_pattern_img(oi_img / f"oi_{i}.jpg", seed=200 + i)
        (oi_lbl / f"oi_{i}.txt").write_text("5 0.5 0.8 0.2 0.15\n", encoding="utf-8")

    # 3. Data lokal beranotasi (2 grup x 2 frame)
    frames_dir = tmp_path / "data/interim/frames"
    ann_dir = tmp_path / "data/interim/annotations"
    manifest_frames = []
    for g in range(2):
        for f in range(2):
            rel = f"almamater/almet_g{g}_f{f}.jpg"
            _make_pattern_img(frames_dir / rel, seed=300 + g * 10 + f)
            lbl_file = (ann_dir / rel).with_suffix(".txt")
            lbl_file.parent.mkdir(parents=True, exist_ok=True)
            lbl_file.write_text("1 0.5 0.4 0.4 0.4\n4 0.5 0.9 0.2 0.1\n", encoding="utf-8")
            manifest_frames.append({"file": rel, "group_id": f"local_almamater_g{g + 1:02d}"})
    (frames_dir / "manifest.json").write_text(
        json.dumps({"total_frames": len(manifest_frames), "frames": manifest_frames}),
        encoding="utf-8",
    )

    manifest = build_detection_dataset(
        version="v_test",
        project_root=tmp_path,
        use_hardlinks=False,
        seed=42,
    )

    assert manifest["total_images"] == 16
    assert manifest["group_leakage_count"] == 0
    assert manifest["sources_collected"]["local"] == 4
    assert manifest["sources_collected"]["fashionpedia"] == 8
    assert manifest["sources_collected"]["openimages"] == 4
    assert manifest["total_instances"]["sandal"] == 4
    assert manifest["total_instances"]["jas_almamater"] >= 4

    data_yaml = Path(manifest["data_yaml"])
    assert data_yaml.is_file()
    loaded = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert loaded["names"][3] == "bawahan_panjang"
    assert (data_yaml.parent / "manifest.json").is_file()


def test_validate_data_yaml_and_draw_preview(tmp_path: Path) -> None:
    cfgs = load_all_configs()
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump({"names": cfgs.classes.id_to_name}), encoding="utf-8")
    _validate_data_yaml(good, cfgs.classes.id_to_name)

    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"names": {0: "wrong"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="tidak cocok"):
        _validate_data_yaml(bad, cfgs.classes.id_to_name)

    canvas = np.zeros((320, 240, 3), dtype=np.uint8)
    det = Detection(
        class_id=1, class_name="jas_almamater", confidence=0.88, bbox=(20, 30, 180, 220)
    )
    drawn = _draw_preview(canvas, [det])
    assert drawn.shape == canvas.shape
    assert int(drawn.sum()) > 0

    dup_same_cls = Detection(
        class_id=1, class_name="jas_almamater", confidence=0.60, bbox=(22, 32, 178, 218)
    )
    diff_cls = Detection(
        class_id=0, class_name="pakaian_berkerah", confidence=0.70, bbox=(22, 32, 178, 218)
    )
    merged = nms_detections([dup_same_cls, det, diff_cls], iou_threshold=0.50)
    assert len(merged) == 2
    assert [m.class_id for m in merged] == [1, 0]


def test_per_source_group_split_preserves_local_in_train(tmp_path: Path) -> None:
    """Pastikan sumber lokal kecil (grup multi-frame) tidak tersedot habis ke val/test."""
    samples: list[DatasetSample] = []
    # 200 gambar publik (1 gambar/grup)
    for i in range(200):
        samples.append(
            DatasetSample(
                sample_id=f"fp_{i}",
                source="fashionpedia",
                group_id=f"fp_{i}",
                image_path=tmp_path / f"fp_{i}.jpg",
                labels=((i % 5, 0.5, 0.5, 0.3, 0.4),),
            )
        )
    # 8 grup lokal meniru ukuran asli Polibatam (61, 18, 10, 10, 8, 5, 3, 3)
    local_sizes = [61, 18, 10, 10, 8, 5, 3, 3]
    for g_idx, size in enumerate(local_sizes):
        for f_idx in range(size):
            samples.append(
                DatasetSample(
                    sample_id=f"loc_g{g_idx}_f{f_idx}",
                    source="local",
                    group_id=f"local_g{g_idx:02d}",
                    image_path=tmp_path / f"loc_{g_idx}_{f_idx}.jpg",
                    labels=((1, 0.5, 0.4, 0.3, 0.4), (4, 0.5, 0.9, 0.1, 0.1)),
                )
            )

    splits = assign_group_splits(samples, num_classes=6, seed=42)
    assert verify_no_group_leakage(splits) == 0

    local_by_split = {
        s: sum(1 for item in items if item.source == "local") for s, items in splits.items()
    }
    assert local_by_split["train"] >= 70
    assert local_by_split["val"] > 0
    assert local_by_split["test"] > 0


def test_import_local_coco_merges_and_cleans(tmp_path: Path) -> None:
    assert (
        strip_roboflow_filename("almet_img_6044_jpg.rf.mDFSzd2JpKdO4SKqLrPv.jpg")
        == "almet_img_6044.jpg"
    )
    assert (
        strip_roboflow_filename("weird.jpg", extra_name="almet_img_5982.jpg")
        == "almet_img_5982.jpg"
    )

    frames_dir = tmp_path / "data/interim/frames"
    _make_pattern_img(frames_dir / "almamater/almet_01.jpg", seed=1)
    _make_pattern_img(frames_dir / "almamater/almet_02.jpg", seed=2)
    _make_pattern_img(frames_dir / "non_almamater/nonalmet_01.jpg", seed=3)
    manifest = {
        "total_frames": 3,
        "frames": [
            {"file": "almamater/almet_01.jpg", "group_id": "local_almamater_g01"},
            {"file": "almamater/almet_02.jpg", "group_id": "local_almamater_g01"},
            {"file": "non_almamater/nonalmet_01.jpg", "group_id": "local_non_almamater_g01"},
        ],
    }
    (frames_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Dua folder COCO terpisah ala Roboflow + folder sementara roboflow_upload
    coco_a = tmp_path / "data/interim/export_a.coco/train"
    coco_b = tmp_path / "data/interim/export_b.coco (1)/train"
    coco_a.mkdir(parents=True)
    coco_b.mkdir(parents=True)
    (tmp_path / "data/interim/roboflow_upload").mkdir(parents=True)

    cats = [
        {"id": 0, "name": "attire-compliance-local", "supercategory": "none"},
        {"id": 1, "name": "bawahan_panjang"},
        {"id": 2, "name": "jas_almamater"},
        {"id": 3, "name": "sandal"},
    ]
    (coco_a / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "categories": cats,
                "images": [
                    {
                        "id": 0,
                        "file_name": "almet_01_jpg.rf.ABC123.jpg",
                        "width": 160,
                        "height": 160,
                        "extra": {"name": "almet_01.jpg"},
                    },
                    {
                        "id": 1,
                        "file_name": "nonalmet_01_jpg.rf.XYZ789.jpg",
                        "width": 160,
                        "height": 160,
                    },
                ],
                "annotations": [
                    {"id": 1, "image_id": 0, "category_id": 2, "bbox": [16, 16, 80, 80]},
                    {"id": 2, "image_id": 1, "category_id": 3, "bbox": [40, 120, 40, 32]},
                ],
            }
        ),
        encoding="utf-8",
    )
    (coco_b / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "categories": cats,
                "images": [
                    {
                        "id": 0,
                        "file_name": "almet_02_jpg.rf.DEF456.jpg",
                        "width": 160,
                        "height": 160,
                        "extra": {"name": "almet_02.jpg"},
                    }
                ],
                "annotations": [
                    {"id": 1, "image_id": 0, "category_id": 1, "bbox": [32, 80, 64, 64]},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = import_local_coco_annotations(
        max_previews=2,
        cleanup_interim=True,
        project_root=tmp_path,
    )
    assert report["annotated_images"] == 3
    assert report["total_annotations"] == 3
    assert report["instances"]["jas_almamater"] == 1
    assert report["instances"]["bawahan_panjang"] == 1
    assert report["instances"]["sandal"] == 1
    assert report["missing_frames"] == []
    assert report["dimension_mismatches"] == []
    assert Path(report["merged_coco_path"]).is_file()

    # Pastikan label YOLO 6-kelas tepat di lokasi yang dibaca build-detection
    lbl_almet1 = tmp_path / "data/interim/annotations/almamater/almet_01.txt"
    boxes, bad = parse_yolo_label_file(lbl_almet1, num_classes=6)
    assert bad == 0
    assert len(boxes) == 1
    assert boxes[0][0] == 1  # jas_almamater == id 1 di configs/classes.yaml
    assert boxes[0][1] == pytest.approx(0.35, abs=1e-4)  # (16 + 40) / 160

    # Folder mentah Roboflow & roboflow_upload sudah dibersihkan
    assert not (tmp_path / "data/interim/export_a.coco").exists()
    assert not (tmp_path / "data/interim/export_b.coco (1)").exists()
    assert not (tmp_path / "data/interim/roboflow_upload").exists()
