"""Unit test untuk validasi konfigurasi YAML."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from attire_compliance.config import ClassesConfig, ClassItem, load_all_configs


def test_load_all_configs_succeeds() -> None:
    cfgs = load_all_configs()
    assert cfgs.classes.names == [
        "pakaian_berkerah",
        "jas_almamater",
        "kaos_oblong",
        "bawahan_panjang",
        "sepatu",
        "sandal",
    ]
    assert cfgs.classes.id_to_name[3] == "bawahan_panjang"
    assert cfgs.detector.imgsz == 640
    assert cfgs.detector.optimizer == "AdamW"
    assert cfgs.detector.lr0 == pytest.approx(0.001)
    assert cfgs.classifier.imgsz == 224
    assert cfgs.classifier.letterbox is True
    assert cfgs.app.fusion.almamater_overrides_tshirt is True
    assert cfgs.app.fusion.mode == "fusion"


def test_classes_config_rejects_wrong_count() -> None:
    with pytest.raises(ValidationError, match="tepat 6"):
        ClassesConfig(
            classes=[
                ClassItem(
                    id=0,
                    name="pakaian_berkerah",
                    display_name="Pakaian Berkerah",
                    description="Kemeja",
                    role="required_top",
                )
            ]
        )


def test_classes_config_rejects_non_contiguous_ids() -> None:
    items = [
        ClassItem(
            id=i if i < 5 else 7,
            name=f"cls_{i}",
            display_name=f"Cls {i}",
            description="desc",
            role="required_top",
        )
        for i in range(6)
    ]
    with pytest.raises(ValidationError, match="berurutan"):
        ClassesConfig(classes=items)


def test_classes_config_rejects_duplicate_names() -> None:
    items = [
        ClassItem(
            id=i,
            name="dup" if i in (0, 1) else f"cls_{i}",
            display_name=f"Cls {i}",
            description="desc",
            role="required_top",
        )
        for i in range(6)
    ]
    with pytest.raises(ValidationError, match="duplikat"):
        ClassesConfig(classes=items)
