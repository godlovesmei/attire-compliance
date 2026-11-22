"""Asosiasi atribut pakaian ke tiap individu `person` berbasis IoA & visibilitas tubuh."""

from __future__ import annotations

from dataclasses import dataclass

from attire_compliance.data.geometry import ioa
from attire_compliance.detection.predictor import Detection


@dataclass(frozen=True)
class PersonBox:
    """Satu orang hasil Stage 0 (YOLOv12 COCO `person` + ByteTrack)."""

    track_id: int
    bbox: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    confidence: float


def associate_attributes_to_persons(
    persons: list[PersonBox],
    detections: list[Detection],
    *,
    min_ioa: float = 0.50,
) -> tuple[dict[int, list[Detection]], list[Detection]]:
    """Assign tiap atribut pakaian ke `PersonBox` dengan IoA tertinggi (>= `min_ioa`)."""
    per_person: dict[int, list[Detection]] = {p.track_id: [] for p in persons}
    unassigned: list[Detection] = []

    if not persons:
        return per_person, list(detections)

    for det in detections:
        best_track: int | None = None
        best_score = 0.0
        for person in persons:
            score = ioa(det.bbox, person.bbox)
            if score >= min_ioa and score > best_score:
                best_score = score
                best_track = person.track_id

        if best_track is not None:
            per_person[best_track].append(det)
        else:
            unassigned.append(det)

    return per_person, unassigned


def is_full_body_visible(
    person_bbox: tuple[float, float, float, float],
    frame_height: int,
    *,
    bottom_edge_margin_px: int = 10,
) -> bool:
    """Cek apakah kaki/bagian bawah tubuh person tidak terpotong tepi bawah frame."""
    if frame_height <= 0:
        return False
    _, _, _, y2 = person_bbox
    return y2 < float(frame_height - bottom_edge_margin_px)
