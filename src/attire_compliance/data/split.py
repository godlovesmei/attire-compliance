"""Deduplikasi perceptual hash lintas sumber & stratified group split anti-leakage."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
from PIL import Image

# Lookup table jumlah bit '1' pada setiap byte (0..255) untuk Hamming distance cepat
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class DatasetSample:
    """Satu gambar beserta seluruh bounding box YOLO 6-kelas dan identitas grup sesinya."""

    sample_id: str
    source: str
    group_id: str
    image_path: Path
    labels: tuple[tuple[int, float, float, float, float], ...]


def compute_phash_uint64(image_path: Path) -> int:
    """Hitung 64-bit perceptual hash (pHash) dari gambar sebagai integer unsigned 64-bit."""
    with Image.open(image_path) as im:
        ph = imagehash.phash(im.convert("RGB"))
    return int(str(ph), 16)


def _hamming_to_array(hash_val: int, existing_hashes: np.ndarray) -> np.ndarray:
    """Hitung Hamming distance antara `hash_val` dan array uint64 `existing_hashes`."""
    xored = np.bitwise_xor(existing_hashes, np.uint64(hash_val))
    byte_view = xored.view(np.uint8).reshape(-1, 8)
    return _POPCOUNT_LUT[byte_view].sum(axis=1)


def deduplicate_samples(
    samples: list[DatasetSample],
    phash_threshold: int = 3,
) -> tuple[list[DatasetSample], list[dict[str, Any]]]:
    """Hapus gambar duplikat/hampir identik lintas sumber berdasarkan pHash Hamming distance.

    Data yang muncul lebih dulu dipertahankan (prioritaskan urutan sumber: lokal → publik).
    """
    if not samples:
        return [], []

    kept: list[DatasetSample] = []
    kept_hashes = np.empty(len(samples), dtype=np.uint64)
    kept_count = 0
    duplicates: list[dict[str, Any]] = []

    for sample in samples:
        h_val = compute_phash_uint64(sample.image_path)
        if kept_count > 0 and phash_threshold >= 0:
            dists = _hamming_to_array(h_val, kept_hashes[:kept_count])
            min_idx = int(np.argmin(dists))
            min_dist = int(dists[min_idx])
            if min_dist <= phash_threshold:
                duplicates.append(
                    {
                        "dropped_id": sample.sample_id,
                        "dropped_source": sample.source,
                        "kept_id": kept[min_idx].sample_id,
                        "kept_source": kept[min_idx].source,
                        "hamming": min_dist,
                    }
                )
                continue

        kept.append(sample)
        kept_hashes[kept_count] = np.uint64(h_val)
        kept_count += 1

    return kept, duplicates


def _assign_single_source_groups(
    samples: list[DatasetSample],
    *,
    num_classes: int,
    ratios: dict[str, float],
    rng: random.Random,
) -> dict[str, list[DatasetSample]]:
    """Stratified group split untuk satu sumber data agar proporsi 70/15/15 terjaga per sumber."""
    split_names = ("train", "val", "test")

    grouped: dict[tuple[str, str], list[DatasetSample]] = {}
    total_class_counts = np.zeros(num_classes, dtype=np.float64)
    for sample in samples:
        grouped.setdefault((sample.source, sample.group_id), []).append(sample)
        for cls_id, *_ in sample.labels:
            if 0 <= cls_id < num_classes:
                total_class_counts[cls_id] += 1.0

    total_images = max(1.0, float(len(samples)))

    group_items = list(grouped.items())
    rng.shuffle(group_items)
    # Tempatkan grup besar (sesi multi-foto/video) terlebih dulu agar stabil
    group_items.sort(key=lambda item: (-len(item[1]), item[0]))

    target_img = {s: ratios[s] * total_images for s in split_names}
    target_cls = {s: ratios[s] * total_class_counts for s in split_names}
    cur_img = {s: 0.0 for s in split_names}
    cur_cls = {s: np.zeros(num_classes, dtype=np.float64) for s in split_names}
    assigned: dict[str, list[DatasetSample]] = {s: [] for s in split_names}

    # Normalisasi relatif per split agar split kecil (val/test 15%) tidak terabaikan
    active_mask = total_class_counts > 0
    num_active = max(1, int(np.sum(active_mask)))
    target_img_safe = {s: max(1.0, target_img[s]) for s in split_names}
    target_cls_safe = {s: np.maximum(1.0, target_cls[s]) for s in split_names}

    for _group_key, group_samples in group_items:
        grp_img = float(len(group_samples))
        grp_cls = np.zeros(num_classes, dtype=np.float64)
        for sample in group_samples:
            for cls_id, *_ in sample.labels:
                if 0 <= cls_id < num_classes:
                    grp_cls[cls_id] += 1.0

        best_splits: list[str] = []
        best_cost: float | None = None

        for split_name in split_names:
            cost = 0.0
            for s in split_names:
                add_i = grp_img if s == split_name else 0.0
                add_c = grp_cls if s == split_name else 0.0
                img_err = ((cur_img[s] + add_i - target_img[s]) / target_img_safe[s]) ** 2
                cls_diff = (cur_cls[s] + add_c - target_cls[s]) / target_cls_safe[s]
                cls_err = float(np.sum((cls_diff[active_mask]) ** 2)) / num_active
                cost += img_err + cls_err
            if best_cost is None or cost < best_cost - 1e-12:
                best_cost = cost
                best_splits = [split_name]
            elif abs(cost - best_cost) <= 1e-12:
                best_splits.append(split_name)

        chosen = best_splits[0] if len(best_splits) == 1 else rng.choice(best_splits)
        assigned[chosen].extend(group_samples)
        cur_img[chosen] += grp_img
        cur_cls[chosen] += grp_cls

    return assigned


def assign_group_splits(
    samples: list[DatasetSample],
    *,
    num_classes: int = 6,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[DatasetSample]]:
    """Bagi sampel ke train/val/test secara stratified per sumber & per grup (zero leakage).

    Setiap sumber (`local`, `fashionpedia`, `openimages`, dst.) dibagi secara independen
    mengikuti rasio 70/15/15 sehingga domain lokal yang berukuran kecil tetap terdistribusi
    secara proporsional di `train`, `val`, dan `test` tanpa group leakage.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Total rasio split harus 1.0, ditemukan {ratio_sum:.4f}")
    if any(r <= 0.0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Setiap rasio train/val/test harus > 0.")

    split_names = ("train", "val", "test")
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    rng = random.Random(seed)

    by_source: dict[str, list[DatasetSample]] = {}
    for sample in samples:
        by_source.setdefault(sample.source, []).append(sample)

    assigned: dict[str, list[DatasetSample]] = {s: [] for s in split_names}
    for source_name in sorted(by_source):
        src_splits = _assign_single_source_groups(
            by_source[source_name],
            num_classes=num_classes,
            ratios=ratios,
            rng=rng,
        )
        for s in split_names:
            assigned[s].extend(src_splits[s])

    return assigned


def verify_no_group_leakage(splits: dict[str, list[DatasetSample]]) -> int:
    """Kembalikan jumlah grup yang muncul di lebih dari satu split (harus 0)."""
    seen: Counter[tuple[str, str]] = Counter()
    for split_samples in splits.values():
        unique_groups = {(s.source, s.group_id) for s in split_samples}
        for grp in unique_groups:
            seen[grp] += 1
    return sum(1 for count in seen.values() if count > 1)
