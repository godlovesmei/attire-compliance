"""Unit test offline untuk modul Fase 2: geometry, Fashionpedia mapping, Roboflow, summary."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from attire_compliance.cli import app
from attire_compliance.config import load_all_configs
from attire_compliance.data.fashionpedia import RawAnnotation, map_fashionpedia_objects
from attire_compliance.data.geometry import (
    clip_bbox,
    expand_box_top,
    ioa,
    is_long_skirt,
    xyxy_to_yolo,
)
from attire_compliance.data.roboflow import import_roboflow_datasets
from attire_compliance.data.summary import collect_raw_data_summary

runner = CliRunner()


def test_geometry_clip_ioa_expand_and_yolo() -> None:
    assert clip_bbox((-10, -5, 50, 100), 200, 400) == (0.0, 0.0, 50.0, 100.0)
    assert clip_bbox((199.5, 10, 220, 50), 200, 400) is None
    assert clip_bbox((10, 10, 11, 50), 200, 400, min_px=2.0) is None

    outer = (100.0, 100.0, 300.0, 300.0)
    inner = (150.0, 110.0, 250.0, 160.0)
    assert ioa(inner, outer) == pytest.approx(1.0)
    assert ioa((0.0, 0.0, 10.0, 10.0), outer) == pytest.approx(0.0)

    expanded = expand_box_top((100.0, 100.0, 200.0, 300.0), top_ratio=0.20, side_ratio=0.05)
    assert expanded == pytest.approx((95.0, 60.0, 205.0, 300.0))

    xc, yc, w, h = xyxy_to_yolo((100.0, 50.0, 300.0, 250.0), img_w=400, img_h=400)
    assert (xc, yc, w, h) == pytest.approx((0.5, 0.375, 0.5, 0.5))


def test_is_long_skirt_distinguishes_long_vs_mini() -> None:
    outfit = [(100.0, 80.0, 250.0, 220.0), (105.0, 220.0, 245.0, 460.0)]
    long_skirt = (105.0, 220.0, 245.0, 460.0)  # h=240 / img_h=500 -> 0.48
    mini_skirt = (105.0, 220.0, 245.0, 310.0)  # h=90 / img_h=500 -> 0.18

    assert is_long_skirt(long_skirt, img_h=500, all_garment_boxes=outfit) is True
    assert is_long_skirt(mini_skirt, img_h=500, all_garment_boxes=[outfit[0], mini_skirt]) is False


def test_map_fashionpedia_collar_lapel_hood_and_skirt_rules() -> None:
    cfgs = load_all_configs()
    fp_cfg = cfgs.sources.fashionpedia
    classes_cfg = cfgs.classes

    raw = [
        # 1. Kemeja + collar -> pakaian_berkerah (0)
        RawAnnotation("shirt, blouse", (100, 100, 300, 300)),
        RawAnnotation("collar", (160, 100, 240, 140)),
        # 2. Polo (top, t-shirt, sweatshirt + collar tanpa hood) -> pakaian_berkerah (0)
        RawAnnotation("top, t-shirt, sweatshirt", (320, 100, 480, 300)),
        RawAnnotation("collar", (370, 100, 430, 135)),
        # 3. Kaos oblong biasa (tanpa collar, tanpa hood) -> kaos_oblong (2)
        RawAnnotation("top, t-shirt, sweatshirt", (500, 100, 650, 300)),
        # 4. Hoodie (top + hood di atas bahu) -> ditolak
        RawAnnotation("top, t-shirt, sweatshirt", (680, 120, 820, 320)),
        RawAnnotation("hood", (710, 90, 790, 145)),
        # 5. Blazer formal (jacket + lapel tanpa hood) -> jas_almamater (1)
        RawAnnotation("jacket", (100, 350, 300, 580)),
        RawAnnotation("lapel", (150, 360, 250, 480)),
        # 6. Jaket hoodie bertopi (jacket + lapel + hood) -> ditolak
        RawAnnotation("jacket", (320, 350, 480, 580)),
        RawAnnotation("lapel", (360, 360, 440, 450)),
        RawAnnotation("hood", (360, 320, 440, 370)),
        # 7. Jaket tanpa lapel -> ditolak
        RawAnnotation("jacket", (500, 350, 650, 580)),
        # 8. Kemeja tanpa collar -> ditolak
        RawAnnotation("shirt, blouse", (680, 350, 820, 550)),
        # 9. Pants & Shoe -> bawahan_panjang (3) & sepatu (4)
        RawAnnotation("pants", (100, 600, 280, 920)),
        RawAnnotation("shoe", (120, 920, 190, 980)),
        # 10. Rok panjang -> bawahan_panjang (3)
        RawAnnotation("skirt", (320, 550, 480, 940)),
        # 11. Rok mini -> ditolak
        RawAnnotation("skirt", (500, 600, 650, 720)),
    ]

    result = map_fashionpedia_objects(
        raw, img_w=1000, img_h=1000, fp_cfg=fp_cfg, classes_cfg=classes_cfg
    )
    mapped_pairs = [(ann.class_name, ann.source_category) for ann in result.annotations]

    assert mapped_pairs == [
        ("pakaian_berkerah", "shirt, blouse"),
        ("pakaian_berkerah", "top, t-shirt, sweatshirt"),
        ("kaos_oblong", "top, t-shirt, sweatshirt"),
        ("jas_almamater", "jacket"),
        ("bawahan_panjang", "pants"),
        ("sepatu", "shoe"),
        ("bawahan_panjang", "skirt"),
    ]
    assert result.rejected_reasons["top, t-shirt, sweatshirt:excluded_by_hood"] == 1
    assert result.rejected_reasons["jacket:excluded_by_hood"] == 1
    assert result.rejected_reasons["jacket:missing_lapel"] == 1
    assert result.rejected_reasons["shirt, blouse:missing_collar"] == 1
    assert result.rejected_reasons["skirt:too_short_for_long_skirt"] == 1


def test_import_roboflow_datasets_zip_and_layouts(tmp_path: Path) -> None:
    rf_root = tmp_path / "roboflow"
    rf_root.mkdir()

    # Dataset 1 sebagai ZIP dengan layout standar Roboflow (`valid/images`)
    zip_src = tmp_path / "staging"
    (zip_src / "train" / "images").mkdir(parents=True)
    (zip_src / "train" / "labels").mkdir(parents=True)
    (zip_src / "valid" / "images").mkdir(parents=True)
    (zip_src / "valid" / "labels").mkdir(parents=True)

    (zip_src / "data.yaml").write_text(
        yaml.safe_dump({"names": ["Shoe", "Sandal", "Boot"]}), encoding="utf-8"
    )
    Image.new("RGB", (64, 64)).save(zip_src / "train" / "images" / "img1.jpg")
    (zip_src / "train" / "labels" / "img1.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.3 0.4 0.2 0.2\n2 0.1 0.1 0.1 0.1\nbad line\n",
        encoding="utf-8",
    )
    Image.new("RGB", (64, 64)).save(zip_src / "valid" / "images" / "img2.jpg")
    (zip_src / "valid" / "labels" / "img2.txt").write_text(
        "1 0.5 0.5 0.3 0.3\n", encoding="utf-8"
    )

    zip_path = rf_root / "shoe-and-sandal.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file_path in zip_src.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(zip_src))

    # Dataset 2 dengan layout `images/train` & unmapped slug
    ds2 = rf_root / "clothes-detection"
    (ds2 / "images" / "train").mkdir(parents=True)
    (ds2 / "labels" / "train").mkdir(parents=True)
    (ds2 / "data.yaml").write_text(
        yaml.safe_dump({"names": {0: "Shirt", 1: "T-Shirt", 2: "Pants"}}),
        encoding="utf-8",
    )
    Image.new("RGB", (64, 64)).save(ds2 / "images" / "train" / "c1.jpg")
    (ds2 / "labels" / "train" / "c1.txt").write_text(
        "0 0.5 0.3 0.4 0.4\n1 0.5 0.3 0.4 0.4\n2 0.5 0.7 0.4 0.5\n",
        encoding="utf-8",
    )

    (rf_root / "unknown-slug").mkdir()

    report = import_roboflow_datasets(roboflow_root_override=rf_root)

    assert "shoe-and-sandal" in report["extracted_zips"]
    assert report["total_images_saved"] == 3
    assert report["total_instances"]["sepatu"] == 1
    assert report["total_instances"]["sandal"] == 2
    assert report["total_instances"]["pakaian_berkerah"] == 1
    assert report["total_instances"]["kaos_oblong"] == 1
    assert report["total_instances"]["bawahan_panjang"] == 1

    ds1_info = report["datasets"]["shoe-and-sandal"]
    assert ds1_info["status"] == "ok"
    assert ds1_info["invalid_lines"] == 1
    assert ds1_info["unmapped_classes"] == {"Boot": 1}
    assert report["datasets"]["unknown-slug"]["status"] == "skipped"

    remapped_path = (
        rf_root
        / "shoe-and-sandal"
        / "_remapped"
        / "labels"
        / "train"
        / "rf_shoe-and-sandal_train_img1.txt"
    )
    remapped_lbl = remapped_path.read_text(encoding="utf-8")
    assert remapped_lbl.splitlines()[0].startswith("4 ")
    assert remapped_lbl.splitlines()[1].startswith("5 ")


def test_collect_raw_data_summary_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for rel, imgs, inst in [
        ("data/raw/fashionpedia", 10, {"pakaian_berkerah": 5, "sepatu": 8}),
        ("data/raw/openimages", 4, {"sandal": 6}),
    ]:
        out_dir = tmp_path / rel
        out_dir.mkdir(parents=True)
        (out_dir / "report.json").write_text(
            json.dumps({"total_images_saved": imgs, "total_instances": inst}),
            encoding="utf-8",
        )

    summary = collect_raw_data_summary(project_root=tmp_path)
    assert summary["total_images"] == 14
    assert summary["total_instances"]["pakaian_berkerah"] == 5
    assert summary["total_instances"]["sandal"] == 6
    assert summary["sources"]["roboflow"]["available"] is False

    monkeypatch.setenv("ATTIRE_PROJECT_ROOT", str(tmp_path))
    res = runner.invoke(app, ["data", "summary"])
    assert res.exit_code == 0, res.stdout
    assert "fashionpedia" in res.stdout
    assert "TOTAL" in res.stdout

    help_res = runner.invoke(app, ["data", "--help"])
    assert help_res.exit_code == 0
    for cmd in ("fetch-fashionpedia", "fetch-openimages", "import-roboflow", "summary"):
        assert cmd in help_res.stdout
