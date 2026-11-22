"""Utilitas geometri bounding box untuk pemrosesan & pemetaan dataset."""

from __future__ import annotations

from collections.abc import Sequence

BBox = tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


def clip_bbox(bbox: BBox, img_w: int, img_h: int, min_px: float = 2.0) -> BBox | None:
    """Potong bounding box xyxy ke batas gambar; kembalikan None jika terlalu kecil."""
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(img_w), float(x1)))
    y1 = max(0.0, min(float(img_h), float(y1)))
    x2 = max(0.0, min(float(img_w), float(x2)))
    y2 = max(0.0, min(float(img_h), float(y2)))
    if (x2 - x1) < min_px or (y2 - y1) < min_px:
        return None
    return (x1, y1, x2, y2)


def box_area(bbox: BBox) -> float:
    """Hitung luas bounding box xyxy."""
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    return w * h


def intersection_area(a: BBox, b: BBox) -> float:
    """Hitung luas irisan dua bounding box xyxy."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def ioa(inner: BBox, container: BBox) -> float:
    """Hitung Intersection over Area terhadap box `inner` (mis. part di dalam garment)."""
    area_inner = box_area(inner)
    if area_inner <= 0.0:
        return 0.0
    return intersection_area(inner, container) / area_inner


def expand_box_top(bbox: BBox, top_ratio: float = 0.20, side_ratio: float = 0.05) -> BBox:
    """Perluas sisi atas & samping box (mis. menangkap `hood` di atas bahu jaket/kaos)."""
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (
        x1 - w * side_ratio,
        y1 - h * top_ratio,
        x2 + w * side_ratio,
        y2,
    )


def xyxy_to_yolo(bbox: BBox, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Konversi bounding box piksel xyxy ke format YOLO ter-normalisasi (xc, yc, w, h)."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Dimensi gambar tidak valid: {img_w}x{img_h}")
    x1, y1, x2, y2 = bbox
    xc = max(0.0, min(1.0, ((x1 + x2) * 0.5) / img_w))
    yc = max(0.0, min(1.0, ((y1 + y2) * 0.5) / img_h))
    w = max(0.0, min(1.0, (x2 - x1) / img_w))
    h = max(0.0, min(1.0, (y2 - y1) / img_h))
    return (xc, yc, w, h)


def is_long_skirt(
    skirt_box: BBox,
    img_h: int,
    all_garment_boxes: Sequence[BBox],
    min_img_height_ratio: float = 0.35,
    min_outfit_span_ratio: float = 0.42,
) -> bool:
    """Saring rok panjang (midi/maxi) vs rok pendek berdasarkan rasio tinggi & outfit span."""
    if img_h <= 0:
        return False
    skirt_h = max(0.0, skirt_box[3] - skirt_box[1])
    if (skirt_h / img_h) < min_img_height_ratio:
        return False

    if all_garment_boxes:
        outfit_ymin = min(b[1] for b in all_garment_boxes)
        outfit_ymax = max(b[3] for b in all_garment_boxes)
        outfit_span = outfit_ymax - outfit_ymin
        if outfit_span > 0.0 and (skirt_h / outfit_span) < min_outfit_span_ratio:
            return False

    return True
