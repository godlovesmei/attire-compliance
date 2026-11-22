"""Preprocessing data lokal: konversi foto (HEIC/JPG), ekstraksi frame video, dedup, grouping."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from attire_compliance.config import load_classes_config
from attire_compliance.paths import resolve_project_path

IMAGE_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}
HEIF_EXTS = {".heic", ".heif"}

_IMG_NUM_RE = re.compile(r"img_(\d+)")
_heif_registered = False


def _register_heif_if_needed(path: Path) -> None:
    """Daftarkan decoder HEIC/HEIF ke Pillow (sekali saja) bila file bertipe HEIC/HEIF."""
    global _heif_registered
    if path.suffix.lower() not in HEIF_EXTS or _heif_registered:
        return
    try:
        import pillow_heif
    except ImportError as exc:
        raise RuntimeError(
            "Library `pillow-heif` belum terpasang. Jalankan: uv sync --extra data"
        ) from exc
    pillow_heif.register_heif_opener()
    _heif_registered = True


def extract_exif_datetime(img: Image.Image) -> str | None:
    """Ambil waktu pengambilan foto dari EXIF sebagai ISO string `YYYY-MM-DDTHH:MM:SS`."""
    try:
        exif = img.getexif()
    except Exception:
        return None

    candidates: list[Any] = []
    try:
        ifd = exif.get_ifd(0x8769)
        candidates.extend([ifd.get(36867), ifd.get(36868)])
    except Exception:
        pass
    candidates.extend([exif.get(36867), exif.get(306)])

    for raw in candidates:
        if isinstance(raw, bytes):
            raw = raw.decode(errors="ignore")
        if not isinstance(raw, str):
            continue
        try:
            return datetime.strptime(raw.strip().strip("\x00"), "%Y:%m:%d %H:%M:%S").isoformat()
        except ValueError:
            continue
    return None


def compute_blur_score(arr: np.ndarray, norm_width: int = 640) -> float:
    """Skor ketajaman (variance of Laplacian) yang dinormalisasi ke lebar `norm_width`."""
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    height, width = gray.shape[:2]
    if width > norm_width:
        scale = norm_width / width
        gray = cv2.resize(
            gray,
            (norm_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_long_edge(img: Image.Image, max_side: int) -> Image.Image:
    """Perkecil sisi terpanjang ke `max_side` (rasio terjaga). Tidak pernah memperbesar."""
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def compute_phash(img: Image.Image) -> Any:
    """Hitung perceptual hash (pHash) gambar untuk deteksi duplikat."""
    import imagehash

    return imagehash.phash(img)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _new_cluster() -> dict[str, Any]:
    return {"members": [], "t_min": None, "t_max": None}


def _add_to_cluster(cluster: dict[str, Any], rec: dict[str, Any]) -> None:
    cluster["members"].append(rec)
    ts = _parse_iso(rec.get("exif_datetime"))
    if ts is None:
        return
    if cluster["t_min"] is None or ts < cluster["t_min"]:
        cluster["t_min"] = ts
    if cluster["t_max"] is None or ts > cluster["t_max"]:
        cluster["t_max"] = ts


def assign_session_groups(
    records: list[dict[str, Any]],
    img_num_gap: int = 50,
    exif_gap_minutes: int = 30,
) -> None:
    """Beri `group_id` sesi pada tiap record agar split train/val/test tidak bocor (leakage).

    Per `batch`: (1) record bernomor `img_NNNN` dikelompokkan berdasarkan kedekatan nomor
    (selisih <= `img_num_gap`), sehingga video dan foto dari sesi yang sama sekelompok;
    (2) record tanpa nomor dikaitkan lewat kedekatan waktu EXIF (<= `exif_gap_minutes`);
    (3) sisanya masuk grup `local_<batch>_misc`.
    """
    gap = timedelta(minutes=exif_gap_minutes)
    by_batch: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_batch.setdefault(rec["batch"], []).append(rec)

    for batch, batch_recs in by_batch.items():
        numbered: list[tuple[int, dict[str, Any]]] = []
        unnumbered: list[dict[str, Any]] = []
        for rec in batch_recs:
            match = _IMG_NUM_RE.search(rec["source_stem"])
            if match:
                numbered.append((int(match.group(1)), rec))
            else:
                unnumbered.append(rec)

        numbered.sort(key=lambda item: (item[0], item[1]["frame_id"]))
        clusters: list[dict[str, Any]] = []
        prev_num: int | None = None
        for num, rec in numbered:
            if prev_num is None or num - prev_num > img_num_gap:
                clusters.append(_new_cluster())
            _add_to_cluster(clusters[-1], rec)
            prev_num = num

        timed = sorted(
            (r for r in unnumbered if _parse_iso(r.get("exif_datetime")) is not None),
            key=lambda r: (r["exif_datetime"], r["frame_id"]),
        )
        misc = [r for r in unnumbered if _parse_iso(r.get("exif_datetime")) is None]

        for rec in timed:
            ts = _parse_iso(rec["exif_datetime"])
            assert ts is not None
            best: dict[str, Any] | None = None
            best_dist: timedelta | None = None
            for cluster in clusters:
                if cluster["t_min"] is None:
                    continue
                if ts < cluster["t_min"]:
                    dist = cluster["t_min"] - ts
                elif ts > cluster["t_max"]:
                    dist = ts - cluster["t_max"]
                else:
                    dist = timedelta(0)
                if dist <= gap and (best_dist is None or dist < best_dist):
                    best, best_dist = cluster, dist
            if best is None:
                best = _new_cluster()
                clusters.append(best)
            _add_to_cluster(best, rec)

        for idx, cluster in enumerate(clusters, start=1):
            group_id = f"local_{batch}_g{idx:02d}"
            for rec in cluster["members"]:
                rec["group_id"] = group_id
        for rec in misc:
            rec["group_id"] = f"local_{batch}_misc"


def _batch_from_path(path: Path, base: Path) -> str:
    parts = path.relative_to(base).parts
    return parts[0] if len(parts) > 1 else "default"


def _clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        elif child.suffix.lower() in {".jpg", ".json", ".txt"}:
            child.unlink()


def _save_jpeg(img: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=quality)


def _make_record(
    *,
    batch: str,
    stem: str,
    file_name: str,
    source_type: str,
    source_rel: str,
    source_stem: str,
    frame_index: int | None,
    img: Image.Image,
    blur_var: float,
    exif_datetime: str | None,
) -> dict[str, Any]:
    return {
        "frame_id": f"{batch}/{stem}",
        "file": f"{batch}/{file_name}",
        "batch": batch,
        "source_type": source_type,
        "source": source_rel,
        "source_stem": source_stem,
        "frame_index": frame_index,
        "width": img.size[0],
        "height": img.size[1],
        "blur_var": round(blur_var, 2),
        "exif_datetime": exif_datetime,
        "group_id": None,
    }


def extract_local_media(
    *,
    local_root: Path | None = None,
    output_dir: Path | None = None,
    configs_dir: Path | None = None,
    max_side: int = 1600,
    jpeg_quality: int = 92,
    video_interval_sec: float = 0.4,
    max_frames_per_video: int = 35,
    min_blur_var: float = 35.0,
    video_phash_threshold: int = 6,
    photo_phash_threshold: int = 4,
) -> dict[str, Any]:
    """Konversi foto lokal ke JPG, ekstrak frame video tajam, dedup, dan beri group_id sesi.

    Data sumber di `local_root` TIDAK pernah diubah; hasil ditulis ke `output_dir`.
    """
    root = local_root if local_root is not None else resolve_project_path("data/raw/local")
    out_dir = output_dir if output_dir is not None else resolve_project_path("data/interim/frames")
    root_res, out_res = root.resolve(), out_dir.resolve()
    if out_res == root_res or root_res in out_res.parents or out_res in root_res.parents:
        raise ValueError(
            f"output_dir tidak boleh berimpit dengan local_root: {out_res} vs {root_res}"
        )

    classes_cfg = load_classes_config(configs_dir)
    _clean_output_dir(out_dir)

    records: list[dict[str, Any]] = []
    skipped_duplicates: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    video_stats: dict[str, dict[str, Any]] = {}
    heic_converted = 0

    # ---- Foto ----
    images_root = root / "images"
    image_paths = (
        sorted(p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if images_root.is_dir()
        else []
    )
    accepted_hashes: dict[str, list[tuple[Any, str]]] = {}
    for path in image_paths:
        batch = _batch_from_path(path, images_root)
        source_rel = path.relative_to(root).as_posix()
        try:
            _register_heif_if_needed(path)
            with Image.open(path) as im:
                exif_dt = extract_exif_datetime(im)  # sebelum exif_transpose
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                rgb.load()
        except Exception as exc:
            failed.append({"source": source_rel, "error": f"{type(exc).__name__}: {exc}"})
            continue

        rgb = resize_long_edge(rgb, max_side)
        phash = compute_phash(rgb)
        stem = path.stem.lower()

        duplicate_of: str | None = None
        hamming = 0
        for prev_hash, prev_id in accepted_hashes.get(batch, []):
            hamming = int(phash - prev_hash)
            if hamming <= photo_phash_threshold:
                duplicate_of = prev_id
                break
        if duplicate_of is not None:
            skipped_duplicates.append(
                {"source": source_rel, "duplicate_of": duplicate_of, "hamming": hamming}
            )
            continue

        blur_var = compute_blur_score(np.asarray(rgb))
        _save_jpeg(rgb, out_dir / batch / f"{stem}.jpg", jpeg_quality)
        record = _make_record(
            batch=batch,
            stem=stem,
            file_name=f"{stem}.jpg",
            source_type="photo",
            source_rel=source_rel,
            source_stem=stem,
            frame_index=None,
            img=rgb,
            blur_var=blur_var,
            exif_datetime=exif_dt,
        )
        records.append(record)
        accepted_hashes.setdefault(batch, []).append((phash, record["frame_id"]))
        if path.suffix.lower() in HEIF_EXTS:
            heic_converted += 1

    # ---- Video ----
    videos_root = root / "videos"
    video_paths = (
        sorted(p for p in videos_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
        if videos_root.is_dir()
        else []
    )
    for path in video_paths:
        batch = _batch_from_path(path, videos_root)
        source_rel = path.relative_to(root).as_posix()
        stem = path.stem.lower()

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            failed.append({"source": source_rel, "error": "VideoCapture gagal membuka video"})
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:
            fps = 30.0
        step = max(1, int(round(fps * video_interval_sec)))

        candidates: list[dict[str, Any]] = []
        idx = 0
        while cap.grab():
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    rgb_arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = resize_long_edge(Image.fromarray(rgb_arr), max_side)
                    candidates.append(
                        {
                            "idx": idx,
                            "blur": compute_blur_score(rgb_arr),
                            "pil": pil,
                            "hash": compute_phash(pil),
                        }
                    )
            idx += 1
        cap.release()

        passed = [c for c in candidates if c["blur"] >= min_blur_var]
        if len(passed) < 5:
            ranked = sorted(candidates, key=lambda c: c["blur"], reverse=True)
            passed = sorted(ranked[: max(5, len(passed))], key=lambda c: c["idx"])
        passed_blur_count = len(passed)

        kept: list[dict[str, Any]] = []
        for cand in passed:
            if all(int(cand["hash"] - k["hash"]) > video_phash_threshold for k in kept):
                kept.append(cand)
        after_dedup = len(kept)

        if len(kept) > max_frames_per_video:
            positions = np.unique(
                np.linspace(0, len(kept) - 1, max_frames_per_video).round().astype(int)
            )
            kept = [kept[int(i)] for i in positions]

        for cand in kept:
            file_stem = f"{stem}_f{cand['idx']:04d}"
            _save_jpeg(cand["pil"], out_dir / batch / f"{file_stem}.jpg", jpeg_quality)
            records.append(
                _make_record(
                    batch=batch,
                    stem=file_stem,
                    file_name=f"{file_stem}.jpg",
                    source_type="video",
                    source_rel=source_rel,
                    source_stem=stem,
                    frame_index=int(cand["idx"]),
                    img=cand["pil"],
                    blur_var=float(cand["blur"]),
                    exif_datetime=None,
                )
            )

        blur_values = [float(c["blur"]) for c in candidates]
        video_stats[source_rel] = {
            "batch": batch,
            "fps": round(float(fps), 2),
            "candidates": len(candidates),
            "passed_blur": passed_blur_count,
            "after_dedup": after_dedup,
            "saved": len(kept),
            "blur_min": round(min(blur_values), 2) if blur_values else None,
            "blur_max": round(max(blur_values), 2) if blur_values else None,
        }

    assign_session_groups(records)

    by_batch: dict[str, dict[str, int]] = {}
    group_counts: dict[str, int] = {}
    for rec in records:
        info = by_batch.setdefault(rec["batch"], {"photos": 0, "video_frames": 0, "total": 0})
        info["photos" if rec["source_type"] == "photo" else "video_frames"] += 1
        info["total"] += 1
        group_counts[rec["group_id"]] = group_counts.get(rec["group_id"], 0) + 1

    report: dict[str, Any] = {
        "local_root": str(root_res),
        "output_dir": str(out_res),
        "max_side": max_side,
        "jpeg_quality": jpeg_quality,
        "total_frames": len(records),
        "by_batch": by_batch,
        "heic_converted": heic_converted,
        "num_skipped_duplicates": len(skipped_duplicates),
        "skipped_duplicates": skipped_duplicates,
        "failed": failed,
        "videos": video_stats,
        "groups": dict(sorted(group_counts.items())),
    }

    (out_dir / "classes.txt").write_text("\n".join(classes_cfg.names) + "\n", encoding="utf-8")
    manifest = {
        "total_frames": len(records),
        "groups": report["groups"],
        "frames": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir.parent / "annotations").mkdir(parents=True, exist_ok=True)
    return report
