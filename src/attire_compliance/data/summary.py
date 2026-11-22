"""Agregasi laporan jumlah gambar & instance per kelas dari seluruh sumber raw."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from attire_compliance.config import load_all_configs
from attire_compliance.paths import resolve_project_path


def _read_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_raw_data_summary(
    *,
    configs_dir: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Kumpulkan ringkasan laporan dari Fashionpedia, Open Images V7, dan Roboflow."""
    cfgs = load_all_configs(configs_dir)
    class_names = cfgs.classes.names

    sources_paths = {
        "fashionpedia": resolve_project_path(cfgs.sources.fashionpedia.output_dir, project_root)
        / "report.json",
        "openimages": resolve_project_path(cfgs.sources.openimages.output_dir, project_root)
        / "report.json",
        "roboflow": resolve_project_path(cfgs.sources.roboflow.root_dir, project_root)
        / "report.json",
    }

    rows: dict[str, dict[str, Any]] = {}
    grand_images = 0
    grand_instances = {name: 0 for name in class_names}

    for src_name, report_path in sources_paths.items():
        data = _read_report(report_path)
        if data is None:
            rows[src_name] = {
                "available": False,
                "images": 0,
                "instances": {name: 0 for name in class_names},
            }
            continue

        images = int(data.get("total_images_saved", 0))
        inst_raw = data.get("total_instances", {})
        instances = {name: int(inst_raw.get(name, 0)) for name in class_names}
        rows[src_name] = {
            "available": True,
            "images": images,
            "instances": instances,
        }
        grand_images += images
        for name, count in instances.items():
            grand_instances[name] += count

    interim_report = _read_report(
        resolve_project_path("data/interim/frames", project_root) / "report.json"
    )
    ann_report = _read_report(
        resolve_project_path("data/interim/annotations", project_root) / "import_report.json"
    )
    local_interim: dict[str, Any] = {
        "available": False,
        "total_frames": 0,
        "groups": 0,
        "by_batch": {},
        "annotated_images": 0,
        "annotated_instances": {name: 0 for name in class_names},
    }
    if interim_report is not None:
        local_interim = {
            "available": True,
            "total_frames": int(interim_report.get("total_frames", 0)),
            "groups": len(interim_report.get("groups", {})),
            "by_batch": interim_report.get("by_batch", {}),
            "annotated_images": int(ann_report.get("annotated_images", 0)) if ann_report else 0,
            "annotated_instances": (
                {name: int(ann_report.get("instances", {}).get(name, 0)) for name in class_names}
                if ann_report
                else {name: 0 for name in class_names}
            ),
        }

    return {
        "class_names": class_names,
        "sources": rows,
        "total_images": grand_images,
        "total_instances": grand_instances,
        "local_interim": local_interim,
    }
