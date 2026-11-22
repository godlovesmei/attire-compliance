"""Unit test untuk asosiasi atribut -> person (IoA) & visibilitas kaki."""

from __future__ import annotations

from attire_compliance.detection.predictor import Detection
from attire_compliance.pipeline.association import (
    PersonBox,
    associate_attributes_to_persons,
    is_full_body_visible,
)


def test_associate_attributes_assigns_to_correct_person_and_tracks_unassigned() -> None:
    p1 = PersonBox(track_id=1, bbox=(50, 20, 250, 460), confidence=0.95)
    p2 = PersonBox(track_id=2, bbox=(320, 20, 520, 460), confidence=0.93)

    d1_shirt = Detection(0, "pakaian_berkerah", 0.85, (80, 100, 220, 240))
    d2_blazer = Detection(1, "jas_almamater", 0.90, (340, 90, 500, 260))
    d_stray = Detection(5, "sandal", 0.70, (580, 400, 620, 450))

    per_person, unassigned = associate_attributes_to_persons(
        [p1, p2], [d1_shirt, d2_blazer, d_stray], min_ioa=0.50
    )

    assert [d.class_name for d in per_person[1]] == ["pakaian_berkerah"]
    assert [d.class_name for d in per_person[2]] == ["jas_almamater"]
    assert [d.class_name for d in unassigned] == ["sandal"]


def test_is_full_body_visible_checks_bottom_margin() -> None:
    assert is_full_body_visible((50, 20, 250, 450), frame_height=480, bottom_edge_margin_px=10)
    assert not is_full_body_visible((50, 20, 250, 475), frame_height=480, bottom_edge_margin_px=10)
