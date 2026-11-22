"""Unit test offline untuk modul Fase 3: preprocessing data lokal (frames.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from typer.testing import CliRunner

from attire_compliance.cli import app
from attire_compliance.data.frames import (
    assign_session_groups,
    compute_blur_score,
    extract_local_media,
    resize_long_edge,
)
from attire_compliance.data.summary import collect_raw_data_summary

runner = CliRunner()


def _pattern(seed: int) -> np.ndarray:
    """Citra RGB sintetis dengan persegi kontras tinggi (tajam & unik per seed)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(6):
        x0, y0 = int(rng.integers(0, 240)), int(rng.integers(0, 160))
        w, h = int(rng.integers(30, 100)), int(rng.integers(30, 80))
        color = tuple(int(c) for c in rng.integers(80, 256, 3))
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), color, -1)
    return img


def _blurry(seed: int) -> np.ndarray:
    return cv2.GaussianBlur(_pattern(seed), (0, 0), 40)


def _write_video(path: Path, rgb_frames: list[np.ndarray], fps: int = 10) -> None:
    height, width = rgb_frames[0].shape[:2]
    if hasattr(cv2.VideoWriter, "fourcc"):
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    else:  # pragma: no cover
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    for frame in rgb_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _rec(batch: str, stem: str, exif: str | None = None) -> dict[str, Any]:
    return {
        "frame_id": f"{batch}/{stem}",
        "batch": batch,
        "source_stem": stem,
        "exif_datetime": exif,
        "group_id": None,
    }


def test_compute_blur_score_sharp_vs_blurry_vs_flat() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 15)
    flat = np.full((240, 320, 3), 128, dtype=np.uint8)

    assert compute_blur_score(sharp) > 1000
    assert compute_blur_score(flat) == 0.0
    assert compute_blur_score(blurred) < compute_blur_score(sharp) / 100


def test_resize_long_edge_keeps_aspect_and_never_upscales() -> None:
    assert resize_long_edge(Image.new("RGB", (2400, 1200)), 1600).size == (1600, 800)
    assert resize_long_edge(Image.new("RGB", (900, 2000)), 1600).size == (720, 1600)
    assert resize_long_edge(Image.new("RGB", (400, 300)), 1600).size == (400, 300)


def test_assign_session_groups_numbers_exif_and_misc() -> None:
    records = [
        _rec("almamater", "almet_img_5989", "2026-11-20T10:00:00"),
        _rec("almamater", "almet_img_5990"),  # frame video #1
        {**_rec("almamater", "almet_img_5990"), "frame_id": "almamater/almet_img_5990_f0100"},
        _rec("almamater", "almet_img_5991"),
        _rec("almamater", "almet_img_6015"),
        _rec("almamater", "almet_img_6044"),
        _rec("almamater", "almet_img_7063"),
        _rec("almamater", "almet_fullsizerender_a", "2026-11-20T10:10:00"),
        _rec("almamater", "almet_fullsizerender_b", "2026-11-21T09:00:00"),
        _rec("almamater", "almet_fullsizerender_c"),
        _rec("non_almamater", "nonalmet_img_0501"),
    ]
    assign_session_groups(records)
    group = {r["frame_id"]: r["group_id"] for r in records}

    same = group["almamater/almet_img_5989"]
    assert group["almamater/almet_img_5990"] == same
    assert group["almamater/almet_img_5990_f0100"] == same
    assert group["almamater/almet_img_5991"] == same
    assert group["almamater/almet_img_6015"] == same
    assert group["almamater/almet_img_6044"] == same
    other_grp = group["almamater/almet_img_7063"]
    assert other_grp != same
    assert group["almamater/almet_fullsizerender_a"] == same
    assert group["almamater/almet_fullsizerender_b"] not in {same, other_grp}
    assert group["almamater/almet_fullsizerender_c"] == "local_almamater_misc"
    assert group["non_almamater/nonalmet_img_0501"].startswith("local_non_almamater_g")


def test_extract_local_media_end_to_end(tmp_path: Path) -> None:
    local = tmp_path / "local"
    (local / "images" / "almamater").mkdir(parents=True)
    (local / "videos" / "almamater").mkdir(parents=True)

    rng = np.random.default_rng(1)
    big = rng.integers(0, 256, (1600, 2400, 3), dtype=np.uint8)
    other = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    Image.fromarray(big).save(local / "images/almamater/almet_img_0001.jpg", quality=95)
    Image.fromarray(other).save(local / "images/almamater/almet_img_0002.png")
    Image.fromarray(big).save(local / "images/almamater/almet_img_0003.jpg", quality=95)  # duplikat

    frames = [
        _pattern(1), _pattern(1), _pattern(2), _pattern(2), _blurry(90),
        _pattern(3), _pattern(3), _pattern(4), _pattern(4), _blurry(91),
        _pattern(5), _pattern(5), _pattern(6), _pattern(6), _blurry(92),
    ]  # fmt: skip
    _write_video(local / "videos/almamater/almet_img_0004.mp4", frames, fps=10)

    out = tmp_path / "frames"
    report = extract_local_media(
        local_root=local,
        output_dir=out,
        video_interval_sec=0.1,
        min_blur_var=35.0,
    )

    assert report["failed"] == []
    assert report["num_skipped_duplicates"] == 1
    assert report["skipped_duplicates"][0]["duplicate_of"] == "almamater/almet_img_0001"
    assert report["by_batch"]["almamater"] == {"photos": 2, "video_frames": 6, "total": 8}
    assert report["total_frames"] == 8
    assert len(report["groups"]) == 1

    saved_big = Image.open(out / "almamater" / "almet_img_0001.jpg")
    assert max(saved_big.size) == 1600
    assert (out / "almamater" / "almet_img_0002.jpg").is_file()
    assert not (out / "almamater" / "almet_img_0003.jpg").exists()
    assert len(list((out / "almamater").glob("almet_img_0004_f*.jpg"))) == 6

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_frames"] == 8
    assert (out / "report.json").is_file()
    class_lines = (out / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert len(class_lines) == 6 and class_lines[0] == "pakaian_berkerah"
    assert (tmp_path / "annotations").is_dir()
    # Data sumber tidak boleh berubah
    assert (local / "images/almamater/almet_img_0003.jpg").is_file()


def test_extract_local_media_video_blur_fallback_never_empty(tmp_path: Path) -> None:
    local = tmp_path / "local"
    (local / "videos" / "almamater").mkdir(parents=True)
    frames: list[np.ndarray] = []
    for seed in range(1, 7):
        frames.extend([_blurry(seed), _blurry(seed)])
    _write_video(local / "videos/almamater/almet_img_0001.mp4", frames, fps=10)

    report = extract_local_media(
        local_root=local,
        output_dir=tmp_path / "frames",
        video_interval_sec=0.1,
        min_blur_var=35.0,
    )
    assert report["total_frames"] >= 1


def test_summary_includes_local_interim_and_cli_help(tmp_path: Path) -> None:
    interim = tmp_path / "data" / "interim" / "frames"
    interim.mkdir(parents=True)
    (interim / "report.json").write_text(
        json.dumps(
            {
                "total_frames": 42,
                "by_batch": {"almamater": {"photos": 40, "video_frames": 2, "total": 42}},
                "groups": {"local_almamater_g01": 30, "local_almamater_g02": 12},
            }
        ),
        encoding="utf-8",
    )
    summary = collect_raw_data_summary(project_root=tmp_path)
    assert summary["local_interim"]["available"] is True
    assert summary["local_interim"]["total_frames"] == 42
    assert summary["local_interim"]["groups"] == 2

    empty = collect_raw_data_summary(project_root=tmp_path / "kosong")
    assert empty["local_interim"]["available"] is False

    res = runner.invoke(app, ["data", "extract-frames", "--help"])
    assert res.exit_code == 0
    assert "--max-side" in res.stdout
