"""Antarmuka CLI utama proyek `attire`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from attire_compliance import __version__
from attire_compliance.classification.train import train_classifier as run_train_classifier
from attire_compliance.config import load_app_config
from attire_compliance.data.classification import build_classification_dataset
from attire_compliance.data.coco_import import import_local_coco_annotations
from attire_compliance.data.fashionpedia import fetch_fashionpedia
from attire_compliance.data.frames import extract_local_media
from attire_compliance.data.merge import build_detection_dataset
from attire_compliance.data.openimages import fetch_openimages
from attire_compliance.data.roboflow import import_roboflow_datasets
from attire_compliance.data.summary import collect_raw_data_summary
from attire_compliance.detection.prelabel import prelabel_local_frames
from attire_compliance.detection.train import train_detector as run_train_detector
from attire_compliance.doctor import run_doctor
from attire_compliance.evaluation.benchmark import run_latency_benchmark
from attire_compliance.evaluation.classifier import evaluate_classifier as run_eval_classifier
from attire_compliance.evaluation.detector import evaluate_detector as run_eval_detector
from attire_compliance.evaluation.end_to_end import ABLATION_MODES, evaluate_end_to_end
from attire_compliance.evaluation.runner import run_full_evaluation
from attire_compliance.paths import resolve_project_path

app = typer.Typer(
    name="attire",
    help="CLI untuk sistem deteksi etika berpakaian mahasiswa (YOLOv12 + EfficientNetV2).",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Perintah pengambilan, ekstraksi, dan pembangunan dataset.")
train_app = typer.Typer(help="Perintah pelatihan model YOLOv12 dan EfficientNetV2-S.")
evaluate_app = typer.Typer(help="Perintah evaluasi metrik dan benchmark performa.")

app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")

console = Console()


@app.command()
def version() -> None:
    """Tampilkan versi paket attire-compliance."""
    console.print(f"attire-compliance {__version__}")


@app.command()
def doctor(
    skip_inference: bool = typer.Option(
        False,
        "--skip-inference",
        help="Lewati smoke test inference YOLOv12 FP16 (hanya cek Python, config, library, CUDA).",
    ),
) -> None:
    """Verifikasi Python, konfigurasi YAML, dependensi, GPU CUDA, dan inference YOLOv12 FP16."""
    results = run_doctor(skip_inference=skip_inference)

    table = Table(title="attire doctor")
    table.add_column("Pemeriksaan", style="bold")
    table.add_column("Status")
    table.add_column("Detail")

    for item in results:
        badge = "[green]OK[/green]" if item.ok else "[red]GAGAL[/red]"
        table.add_row(item.name, badge, item.detail)

    console.print(table)
    if not all(item.ok for item in results):
        raise typer.Exit(code=1)


@data_app.command("fetch-fashionpedia")
def data_fetch_fashionpedia(
    max_train: int = typer.Option(
        2000, "--max-train", min=0, help="Maksimum gambar train (0 = semua)."
    ),
    max_val: int = typer.Option(
        500, "--max-val", min=0, help="Maksimum gambar validasi (0 = semua)."
    ),
    streaming: bool = typer.Option(
        True,
        "--streaming/--no-streaming",
        help="Gunakan mode streaming Hugging Face agar hemat disk & bandwidth.",
    ),
) -> None:
    """Tarik & filter dataset Fashionpedia dari Hugging Face."""
    console.print(
        f"[cyan]Menarik Fashionpedia (max_train={max_train}, max_val={max_val}, "
        f"streaming={streaming})...[/cyan]"
    )
    report = fetch_fashionpedia(max_train=max_train, max_val=max_val, streaming=streaming)

    split_table = Table(title="Fashionpedia — Ringkasan Split")
    split_table.add_column("Split", style="bold")
    split_table.add_column("Scanned", justify="right")
    split_table.add_column("Saved", justify="right")
    split_table.add_column("Skipped Empty", justify="right")
    for split_name, info in report["splits"].items():
        split_table.add_row(
            split_name,
            str(info["scanned"]),
            str(info["saved"]),
            str(info["skipped_empty"]),
        )
    console.print(split_table)

    cls_table = Table(title="Fashionpedia — Instance per Kelas")
    cls_table.add_column("Kelas Target", style="bold")
    cls_table.add_column("Train", justify="right")
    cls_table.add_column("Val", justify="right")
    cls_table.add_column("Total", justify="right")
    train_inst = report["splits"].get("train", {}).get("instances", {})
    val_inst = report["splits"].get("val", {}).get("instances", {})
    for cls_name, total in report["total_instances"].items():
        cls_table.add_row(
            cls_name,
            str(train_inst.get(cls_name, 0)),
            str(val_inst.get(cls_name, 0)),
            str(total),
        )
    console.print(cls_table)

    rej_table = Table(title="Fashionpedia — Filter Penolakan (Quality Gate)")
    rej_table.add_column("Alasan Penolakan", style="bold")
    rej_table.add_column("Train", justify="right")
    rej_table.add_column("Val", justify="right")
    train_rej = report["splits"].get("train", {}).get("rejected_reasons", {})
    val_rej = report["splits"].get("val", {}).get("rejected_reasons", {})
    all_reasons = sorted(set(train_rej) | set(val_rej))
    if all_reasons:
        for reason in all_reasons:
            rej_table.add_row(reason, str(train_rej.get(reason, 0)), str(val_rej.get(reason, 0)))
        console.print(rej_table)


@data_app.command("fetch-openimages")
def data_fetch_openimages(
    classes: Annotated[
        list[str] | None,
        typer.Option(
            "--classes",
            help="Kelas Open Images V7 yang ditarik (repeatable). Default: Sandal.",
        ),
    ] = None,
    max_train: Annotated[
        int | None, typer.Option("--max-train", min=0, help="Batas sampel train.")
    ] = None,
    max_val: Annotated[
        int | None, typer.Option("--max-val", min=0, help="Batas sampel val.")
    ] = None,
    max_test: Annotated[
        int | None, typer.Option("--max-test", min=0, help="Batas sampel test.")
    ] = None,
) -> None:
    """Tarik kelas pelengkap Open Images V7 menggunakan FiftyOne."""
    selected = classes if classes else ["Sandal"]
    console.print(f"[cyan]Menarik Open Images V7 untuk kelas: {', '.join(selected)}...[/cyan]")
    report = fetch_openimages(
        classes=selected,
        max_train=max_train,
        max_val=max_val,
        max_test=max_test,
    )

    table = Table(title="Open Images V7 — Ringkasan")
    table.add_column("Split", style="bold")
    table.add_column("Saved", justify="right")
    table.add_column("Skipped Empty", justify="right")
    table.add_column("Total Instances", justify="right")
    for split_name, info in report["splits"].items():
        inst_sum = sum(info["instances"].values())
        table.add_row(split_name, str(info["saved"]), str(info["skipped_empty"]), str(inst_sum))
    console.print(table)
    console.print(f"[yellow]Catatan: {report['partial_annotation_warning']}[/yellow]")


@data_app.command("import-roboflow")
def data_import_roboflow(
    slug: str | None = typer.Option(
        None, "--slug", help="Nama folder dataset spesifik di data/raw/roboflow/."
    ),
) -> None:
    """Ekstrak ZIP dan remap dataset Roboflow Universe ke 6 kelas terkunci."""
    report = import_roboflow_datasets(slug=slug)
    if report["extracted_zips"]:
        console.print(
            f"[green]ZIP diekstrak: {', '.join(report['extracted_zips'])}[/green]"
        )

    table = Table(title="Roboflow Universe — Ringkasan Remap")
    table.add_column("Dataset Slug", style="bold")
    table.add_column("Status")
    table.add_column("Saved", justify="right")
    table.add_column("Skipped / Invalid", justify="right")
    table.add_column("Detail")
    for ds_slug, info in report["datasets"].items():
        if info["status"] == "ok":
            total_inst = sum(info["instances"].values())
            unmapped = ", ".join(info["unmapped_classes"].keys()) or "-"
            table.add_row(
                ds_slug,
                "[green]OK[/green]",
                str(info["saved"]),
                f"{info['skipped_empty']} / {info['invalid_lines']}",
                f"instances={total_inst}, unmapped={unmapped}",
            )
        else:
            table.add_row(ds_slug, "[yellow]SKIP[/yellow]", "0", "-", info.get("reason", ""))
    console.print(table)


@data_app.command("summary")
def data_summary() -> None:
    """Tampilkan rekap jumlah gambar & instance per kelas dari seluruh sumber raw."""
    summary = collect_raw_data_summary()
    class_names: list[str] = summary["class_names"]

    table = Table(title="Rekap Dataset Raw Publik")
    table.add_column("Sumber", style="bold", no_wrap=True, min_width=12)
    table.add_column("Status", no_wrap=True, min_width=9)
    table.add_column("Gambar", justify="right", no_wrap=True, min_width=6)
    for cls_name in class_names:
        parts = cls_name.split("_")
        header = "\n".join(parts)
        table.add_column(
            header,
            justify="right",
            no_wrap=True,
            min_width=max(len(p) for p in parts),
        )

    for src_name, info in summary["sources"].items():
        status = "[green]siap[/green]" if info["available"] else "[dim]belum ada[/dim]"
        row = [src_name, status, str(info["images"])]
        row.extend(str(info["instances"].get(c, 0)) for c in class_names)
        table.add_row(*row)

    total_row = ["TOTAL", "-", str(summary["total_images"])]
    total_row.extend(str(summary["total_instances"].get(c, 0)) for c in class_names)
    table.add_row(*total_row, style="bold cyan")
    console.print(table)

    interim = summary.get("local_interim", {})
    if interim.get("available"):
        ann_count = interim.get("annotated_images", 0)
        ann_total = sum(interim.get("annotated_instances", {}).values())
        console.print(
            f"[cyan]Data lokal: {interim['total_frames']} frame dalam {interim['groups']} "
            f"grup sesi (anotasi terverifikasi: {ann_count} frame, {ann_total} box "
            f"di data/interim/annotations/).[/cyan]"
        )


@data_app.command("import-local-coco")
def data_import_local_coco(
    source: Annotated[
        list[Path] | None,
        typer.Option(
            "--source",
            help="Folder/file ekspor COCO Roboflow (repeatable). Kosongkan untuk auto-detect.",
        ),
    ] = None,
    max_previews: Annotated[
        int, typer.Option("--max-previews", min=0, help="Jumlah preview overlay verifikasi.")
    ] = 8,
    cleanup: Annotated[
        bool,
        typer.Option(
            "--cleanup/--no-cleanup",
            help="Hapus folder ekspor mentah Roboflow & prelabel v0 lama setelah tervalidasi.",
        ),
    ] = False,
) -> None:
    """Gabungkan ekspor COCO Roboflow data lokal dan konversi ke label YOLO terverifikasi."""
    console.print("[cyan]Menggabungkan & memvalidasi anotasi COCO lokal → YOLO...[/cyan]")
    report = import_local_coco_annotations(
        sources=source,
        max_previews=max_previews,
        cleanup_interim=cleanup,
    )

    src_table = Table(title="Sumber Ekspor COCO Digabungkan")
    src_table.add_column("Folder / File", style="bold")
    src_table.add_column("JSON", justify="right")
    src_table.add_column("Gambar", justify="right")
    src_table.add_column("Box Valid", justify="right")
    for item in report["sources"]:
        src_table.add_row(
            item["source"],
            str(item["json_files"]),
            str(item["images"]),
            str(item["annotations"]),
        )
    console.print(src_table)

    cls_table = Table(title="Anotasi Lokal Terverifikasi — Instance per Kelas")
    cls_table.add_column("Kelas", style="bold")
    cls_table.add_column("Total Box", justify="right")
    for cls_name, cnt in report["instances"].items():
        cls_table.add_row(cls_name, str(cnt))
    cls_table.add_row("TOTAL", str(report["total_annotations"]), style="bold cyan")
    console.print(cls_table)

    console.print(
        f"[green]Berhasil menggabungkan {report['annotated_images']}/"
        f"{report['total_manifest_frames']} frame ke {report['annotations_dir']}[/green]\n"
        f"Arsip COCO gabungan: {report['merged_coco_path']} | "
        f"Overlap: {len(report['overlaps'])} | Dim mismatch: {len(report['dimension_mismatches'])}"
    )
    if report["missing_frames"]:
        console.print(
            f"[yellow]Frame tanpa anotasi ({len(report['missing_frames'])}): "
            f"{', '.join(report['missing_frames'])}[/yellow]"
        )
    if report["cleaned_paths"]:
        console.print(
            f"[green]Folder/file sementara dirapikan: {', '.join(report['cleaned_paths'])}[/green]"
        )


@data_app.command("extract-frames")
def data_extract_frames(
    max_side: Annotated[
        int, typer.Option("--max-side", min=256, help="Sisi terpanjang maksimum (piksel).")
    ] = 1600,
    video_interval_sec: Annotated[
        float,
        typer.Option("--video-interval-sec", min=0.05, help="Jarak sampling kandidat frame video."),
    ] = 0.4,
    max_frames_per_video: Annotated[
        int,
        typer.Option("--max-frames-per-video", min=1, help="Batas frame tersimpan per video."),
    ] = 35,
    min_blur_var: Annotated[
        float,
        typer.Option("--min-blur-var", min=0.0, help="Ambang minimum ketajaman frame video."),
    ] = 35.0,
    video_phash_threshold: Annotated[
        int,
        typer.Option("--video-phash-threshold", min=0, max=64, help="Hamming dedup frame video."),
    ] = 6,
    photo_phash_threshold: Annotated[
        int,
        typer.Option("--photo-phash-threshold", min=0, max=64, help="Hamming dedup foto."),
    ] = 4,
) -> None:
    """Konversi foto lokal (HEIC→JPG) & ekstrak frame tajam dari video lokal."""
    console.print("[cyan]Memproses data lokal (data/raw/local → data/interim/frames)...[/cyan]")
    report = extract_local_media(
        max_side=max_side,
        video_interval_sec=video_interval_sec,
        max_frames_per_video=max_frames_per_video,
        min_blur_var=min_blur_var,
        video_phash_threshold=video_phash_threshold,
        photo_phash_threshold=photo_phash_threshold,
    )

    batch_table = Table(title="Data Lokal — Ringkasan per Batch")
    batch_table.add_column("Batch", style="bold")
    batch_table.add_column("Foto", justify="right")
    batch_table.add_column("Frame Video", justify="right")
    batch_table.add_column("Total", justify="right")
    for batch_name, info in sorted(report["by_batch"].items()):
        batch_table.add_row(
            batch_name, str(info["photos"]), str(info["video_frames"]), str(info["total"])
        )
    batch_table.add_row(
        "TOTAL",
        str(sum(i["photos"] for i in report["by_batch"].values())),
        str(sum(i["video_frames"] for i in report["by_batch"].values())),
        str(report["total_frames"]),
        style="bold cyan",
    )
    console.print(batch_table)
    console.print(
        f"HEIC dikonversi: {report['heic_converted']} | "
        f"Duplikat dilewati: {report['num_skipped_duplicates']} | "
        f"Gagal: {len(report['failed'])}"
    )
    for item in report["failed"]:
        console.print(f"[red]GAGAL: {item['source']} — {item['error']}[/red]")

    video_table = Table(title="Data Lokal — Ekstraksi Video")
    video_table.add_column("Video", style="bold")
    video_table.add_column("Kandidat", justify="right")
    video_table.add_column("Lolos blur", justify="right")
    video_table.add_column("Setelah dedup", justify="right")
    video_table.add_column("Tersimpan", justify="right")
    for source, info in report["videos"].items():
        video_table.add_row(
            source,
            str(info["candidates"]),
            str(info["passed_blur"]),
            str(info["after_dedup"]),
            str(info["saved"]),
        )
    if report["videos"]:
        console.print(video_table)

    group_table = Table(title="Data Lokal — Grup Sesi (untuk group split Fase 4)")
    group_table.add_column("group_id", style="bold")
    group_table.add_column("Jumlah Frame", justify="right")
    for group_id, count in report["groups"].items():
        group_table.add_row(group_id, str(count))
    console.print(group_table)


@data_app.command("build-detection")
def data_build_detection(
    version: Annotated[
        str, typer.Option("--version", help="Versi dataset output (mis. v0, v1).")
    ] = "v1",
    include_fashionpedia: Annotated[
        bool, typer.Option("--include-fashionpedia/--no-include-fashionpedia")
    ] = True,
    include_openimages: Annotated[
        bool, typer.Option("--include-openimages/--no-include-openimages")
    ] = True,
    include_roboflow: Annotated[
        bool, typer.Option("--include-roboflow/--no-include-roboflow")
    ] = True,
    include_local: Annotated[
        bool, typer.Option("--include-local/--no-include-local")
    ] = True,
    annotations_dir: Annotated[
        Path | None,
        typer.Option("--annotations-dir", help="Direktori label YOLO lokal (opsional)."),
    ] = None,
    phash_threshold: Annotated[
        int, typer.Option("--phash-threshold", min=0, max=64, help="Batas Hamming dedup lintas.")
    ] = 3,
    seed: Annotated[int, typer.Option("--seed", help="Seed random group split.")] = 42,
    use_hardlinks: Annotated[
        bool, typer.Option("--hardlinks/--copy", help="Hardlink NTFS (cepat) atau salin gambar.")
    ] = True,
) -> None:
    """Gabungkan, validasi, dedup pHash, dan group-split dataset deteksi (70/15/15)."""
    console.print(f"[cyan]Membangun dataset deteksi `{version}` (seed={seed})...[/cyan]")
    manifest = build_detection_dataset(
        version=version,
        include_fashionpedia=include_fashionpedia,
        include_openimages=include_openimages,
        include_roboflow=include_roboflow,
        include_local=include_local,
        annotations_dir=annotations_dir,
        phash_threshold=phash_threshold,
        seed=seed,
        use_hardlinks=use_hardlinks,
    )

    console.print(
        f"[green]Dataset `{version}` siap di {manifest['output_dir']}[/green]\n"
        f"Total terkumpul: {manifest['total_collected']} | "
        f"Duplikat dibuang: {manifest['duplicates_removed']} | "
        f"Total bersih: {manifest['total_images']} | "
        f"Group leakage: {manifest['group_leakage_count']}"
    )

    split_table = Table(title=f"Dataset Deteksi {version} — Distribusi Split & Kelas")
    split_table.add_column("Split", style="bold", no_wrap=True)
    split_table.add_column("Gambar", justify="right", no_wrap=True)
    split_table.add_column("Grup", justify="right", no_wrap=True)
    class_names = list(manifest["total_instances"].keys())
    for cls_name in class_names:
        parts = cls_name.split("_")
        split_table.add_column(
            "\n".join(parts),
            justify="right",
            no_wrap=True,
            min_width=max(len(p) for p in parts),
        )

    for split_name in ("train", "val", "test"):
        info = manifest["splits"][split_name]
        row = [split_name, str(info["images"]), str(info["groups"])]
        row.extend(str(info["instances"].get(c, 0)) for c in class_names)
        split_table.add_row(*row)

    total_row = ["TOTAL", str(manifest["total_images"]), "-"]
    total_row.extend(str(manifest["total_instances"].get(c, 0)) for c in class_names)
    split_table.add_row(*total_row, style="bold cyan")
    console.print(split_table)


@data_app.command("prelabel-local")
def data_prelabel_local(
    weights: Annotated[
        Path,
        typer.Option("--weights", help="Path checkpoint detektor (mis. yolo12n_attire_v0.pt)."),
    ],
    conf: Annotated[
        float, typer.Option("--conf", min=0.05, max=1.0, help="Confidence threshold pre-label.")
    ] = 0.25,
    iou: Annotated[
        float, typer.Option("--iou", min=0.1, max=1.0, help="IoU NMS threshold.")
    ] = 0.50,
    max_previews: Annotated[
        int, typer.Option("--max-previews", min=0, help="Jumlah gambar preview overlay disimpan.")
    ] = 12,
    person_crops: Annotated[
        bool,
        typer.Option(
            "--person-crops/--full-frame",
            help="Deteksi per crop `person` (Stage 0) agar foto grup multi-orang terdeteksi.",
        ),
    ] = True,
    device: Annotated[str, typer.Option("--device", help="Device inferensi (0 / cpu).")] = "0",
) -> None:
    """Jalankan pre-annotation otomatis pada `data/interim/frames/` menggunakan model v0."""
    console.print(
        f"[cyan]Menjalankan pre-labeling lokal dengan {weights} "
        f"(conf={conf}, person_crops={person_crops})...[/cyan]"
    )
    report = prelabel_local_frames(
        weights_path=weights,
        conf=conf,
        iou=iou,
        max_previews=max_previews,
        use_person_crops=person_crops,
        device=device,
    )

    table = Table(title="Pre-Annotation Lokal — Ringkasan Deteksi")
    table.add_column("Kelas", style="bold")
    table.add_column("Total Box", justify="right")
    table.add_column("Mean Conf", justify="right")
    for cls_name, count in report["instances"].items():
        table.add_row(cls_name, str(count), f"{report['mean_confidence'][cls_name]:.3f}")
    console.print(table)
    console.print(
        f"[green]Pre-label disimpan ke {report['prelabels_dir']}[/green]\n"
        f"Frame terdeteksi: {report['frames_with_detections']}/{report['total_frames']} | "
        f"Preview visual: {len(report['preview_files'])} file di {report['previews_dir']}"
    )


@data_app.command("build-classification")
def data_build_classification(
    version: Annotated[
        str, typer.Option("--version", help="Versi output dataset classifier (mis. v1).")
    ] = "v1",
    detection_version: Annotated[
        str | None,
        typer.Option("--detection-version", help="Versi dataset deteksi sumber (default=version)."),
    ] = None,
    person_conf: Annotated[
        float, typer.Option("--person-conf", min=0.1, max=1.0)
    ] = 0.45,
    device: Annotated[str, typer.Option("--device")] = "0",
) -> None:
    """Bangun dataset crop person full-body (sesuai/tidak_sesuai) untuk EfficientNetV2-S."""
    console.print(
        f"[cyan]Membangun dataset klasifikasi `{version}` dari deteksi "
        f"`{detection_version or version}`...[/cyan]"
    )
    manifest = build_classification_dataset(
        version=version,
        detection_version=detection_version,
        person_conf=person_conf,
        device=device,
    )
    table = Table(title=f"Dataset Klasifikasi {version} — Ringkasan Crop Person")
    table.add_column("Split", style="bold")
    table.add_column("sesuai", justify="right")
    table.add_column("tidak_sesuai", justify="right")
    table.add_column("Total", justify="right")
    for split_name in ("train", "val", "test"):
        info = manifest["splits"][split_name]
        s_cnt = info.get("sesuai", 0)
        t_cnt = info.get("tidak_sesuai", 0)
        table.add_row(split_name, str(s_cnt), str(t_cnt), str(s_cnt + t_cnt))
    tot = manifest["total_by_class"]
    table.add_row(
        "TOTAL",
        str(tot.get("sesuai", 0)),
        str(tot.get("tidak_sesuai", 0)),
        str(manifest["total_crops"]),
        style="bold cyan",
    )
    console.print(table)
    console.print(f"[green]Review queue CSV: {manifest['review_queue_csv']}[/green]")


@train_app.command("detector")
def train_detector(
    data_yaml: Annotated[
        Path | None,
        typer.Option("--data-yaml", help="Override path data.yaml (default dari detector.yaml)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Override arsitektur/bobot (mis. yolo12n.pt / yolo12s.pt)."),
    ] = None,
    run_name: Annotated[
        str | None, typer.Option("--run-name", help="Nama run (mis. yolo12n_attire_v0).")
    ] = None,
    output_weights: Annotated[
        Path | None, typer.Option("--output-weights", help="Path output checkpoint terbaik.")
    ] = None,
    epochs: Annotated[int | None, typer.Option("--epochs", min=1)] = None,
    batch: Annotated[int | None, typer.Option("--batch", min=1)] = None,
    imgsz: Annotated[int | None, typer.Option("--imgsz", min=128)] = None,
    patience: Annotated[int | None, typer.Option("--patience", min=0)] = None,
    workers: Annotated[int | None, typer.Option("--workers", min=0)] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Latih detektor 6 atribut pakaian YOLOv12."""
    console.print(
        f"[cyan]Memulai training detektor YOLOv12 (run={run_name or 'default'}, "
        f"epochs={epochs or 'config'})...[/cyan]"
    )
    summary = run_train_detector(
        data_yaml=data_yaml,
        model_name=model,
        run_name=run_name,
        output_weights=output_weights,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        workers=workers,
        device=device,
    )

    table = Table(title=f"Hasil Training Detektor — {summary['run_name']}")
    table.add_column("Metrik", style="bold")
    table.add_column("Nilai", justify="right")
    for key, val in summary["metrics"].items():
        table.add_row(key, f"{val:.4f}")
    console.print(table)

    if summary["per_class_ap50"]:
        cls_table = Table(title="AP@50 per Kelas (Val)")
        cls_table.add_column("Kelas", style="bold")
        cls_table.add_column("AP@50", justify="right")
        for cls_name, ap in summary["per_class_ap50"].items():
            cls_table.add_row(cls_name, f"{ap:.4f}")
        console.print(cls_table)

    console.print(f"[green]Bobot terbaik disalin ke: {summary['best_weights']}[/green]")


@train_app.command("classifier")
def train_classifier(
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Override folder dataset classifier.")
    ] = None,
    checkpoint_path: Annotated[
        Path | None, typer.Option("--checkpoint-path", help="Override path checkpoint .pt.")
    ] = None,
    run_name: Annotated[
        str, typer.Option("--run-name", help="Nama run training classifier.")
    ] = "effnetv2s_compliance_v1",
    epochs: Annotated[int | None, typer.Option("--epochs", min=1)] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", min=1)] = None,
    lr: Annotated[float | None, typer.Option("--lr", min=1e-7)] = None,
    patience: Annotated[int | None, typer.Option("--patience", min=0)] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Latih classifier kepatuhan berpakaian EfficientNetV2-S."""
    console.print(f"[cyan]Memulai training EfficientNetV2-S (run={run_name})...[/cyan]")
    summary = run_train_classifier(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        device_override=device,
    )
    table = Table(title=f"Hasil Training Classifier — {summary['run_name']}")
    table.add_column("Metrik (Val Terbaik)", style="bold")
    table.add_column("Nilai", justify="right")
    for key in ("accuracy", "precision_sesuai", "recall_sesuai", "f1_sesuai", "macro_f1"):
        table.add_row(key, f"{summary['val_metrics'].get(key, 0.0):.4f}")
    console.print(table)
    console.print(f"[green]Checkpoint terbaik disimpan ke: {summary['checkpoint_path']}[/green]")


@evaluate_app.command("detector")
def evaluate_detector_cmd(
    weights: Annotated[Path | None, typer.Option("--weights")] = None,
    data_yaml: Annotated[Path | None, typer.Option("--data-yaml")] = None,
    split: Annotated[str, typer.Option("--split")] = "test",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    device: Annotated[str, typer.Option("--device")] = "0",
) -> None:
    """Evaluasi detektor YOLOv12 pada test set (overall + subset kampus lokal)."""
    out = output_dir or resolve_project_path("outputs/reports/detector_eval")
    report = run_eval_detector(
        weights_path=weights,
        data_yaml=data_yaml,
        split=split,
        output_dir=out,
        device=device,
    )
    ov = report["overall"]
    console.print(
        f"[green]Evaluasi Detektor ({split}): P={ov['precision']:.4f}, R={ov['recall']:.4f}, "
        f"mAP@50={ov['mAP50']:.4f}, mAP@50-95={ov['mAP50-95']:.4f}[/green]"
    )


@evaluate_app.command("classifier")
def evaluate_classifier_cmd(
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    data_dir: Annotated[Path | None, typer.Option("--data-dir")] = None,
    split: Annotated[str, typer.Option("--split")] = "test",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    device: Annotated[str, typer.Option("--device")] = "cuda",
) -> None:
    """Evaluasi classifier EfficientNetV2-S pada test set (CM 2x2 + ROC-AUC)."""
    out = output_dir or resolve_project_path("outputs/reports/classifier_eval")
    report = run_eval_classifier(
        checkpoint_path=checkpoint,
        data_dir=data_dir,
        split=split,
        output_dir=out,
        device=device,
    )
    m = report["metrics"]
    console.print(
        f"[green]Evaluasi Classifier ({split}): Acc={m['accuracy']:.4f}, "
        f"Macro-F1={m['macro_f1']:.4f}, ROC-AUC={m['roc_auc']}[/green]"
    )


@evaluate_app.command("end-to-end")
def evaluate_end_to_end_cmd(
    detector_weights: Annotated[Path | None, typer.Option("--detector-weights")] = None,
    classifier_weights: Annotated[Path | None, typer.Option("--classifier-weights")] = None,
    detection_dir: Annotated[Path | None, typer.Option("--detection-dir")] = None,
    split: Annotated[str, typer.Option("--split")] = "test",
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    max_images: Annotated[int | None, typer.Option("--max-images", min=1)] = None,
) -> None:
    """Evaluasi end-to-end pipeline & tabel ablation (rule_only vs classifier_only vs fusion)."""
    from attire_compliance.pipeline.pipeline import AttireCompliancePipeline

    out = output_dir or resolve_project_path("outputs/reports/end_to_end_eval")
    pipeline = AttireCompliancePipeline(
        attribute_weights=detector_weights,
        classifier_weights=classifier_weights,
    )
    pipeline.warmup()
    report = evaluate_end_to_end(
        pipeline,
        detection_dir=detection_dir,
        split=split,
        output_dir=out,
        max_images=max_images,
    )
    table = Table(title=f"Ablation End-to-End ({split}, n={report['evaluated_persons']})")
    table.add_column("Mode", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1 Sesuai", justify="right")
    table.add_column("Recall Tidak Sesuai", justify="right")
    table.add_column("Macro-F1", justify="right")
    for mode in ABLATION_MODES:
        m = report["ablation_overall"][mode]
        table.add_row(
            mode,
            f"{m['accuracy']:.4f}",
            f"{m['f1_sesuai']:.4f}",
            f"{m['recall_tidak_sesuai']:.4f}",
            f"{m['macro_f1']:.4f}",
        )
    console.print(table)


@evaluate_app.command("benchmark")
def evaluate_benchmark_cmd(
    sample_dir: Annotated[Path | None, typer.Option("--sample-dir")] = None,
    detector_weights: Annotated[Path | None, typer.Option("--detector-weights")] = None,
    classifier_weights: Annotated[Path | None, typer.Option("--classifier-weights")] = None,
    warmup_frames: Annotated[int, typer.Option("--warmup", min=1)] = 20,
    measure_frames: Annotated[int, typer.Option("--frames", min=10)] = 300,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Benchmark latensi per stage (mean/p95) & FPS end-to-end."""
    from attire_compliance.pipeline.pipeline import AttireCompliancePipeline

    out = output_dir or resolve_project_path("outputs/reports/benchmark_eval")
    src_dir = sample_dir or resolve_project_path("data/processed/detection/v1/images/test")
    pipeline = AttireCompliancePipeline(
        attribute_weights=detector_weights,
        classifier_weights=classifier_weights,
    )
    pipeline.warmup()
    report = run_latency_benchmark(
        pipeline,
        sample_dir=src_dir,
        output_dir=out,
        warmup_frames=warmup_frames,
        measure_frames=measure_frames,
    )
    console.print(
        f"[green]Benchmark ({report['measure_frames']} frame): "
        f"{report['fps_mean']} FPS (mean={report['stages']['total_ms']['mean_ms']} ms, "
        f"p95={report['stages']['total_ms']['p95_ms']} ms)[/green]"
    )


@evaluate_app.command("all")
def evaluate_all(
    tag: Annotated[str, typer.Option("--tag", help="Label folder laporan Bab 4.")] = "v1_bab4",
    detector_weights: Annotated[Path | None, typer.Option("--detector-weights")] = None,
    classifier_weights: Annotated[Path | None, typer.Option("--classifier-weights")] = None,
    split: Annotated[str, typer.Option("--split")] = "test",
    warmup_frames: Annotated[int, typer.Option("--warmup", min=1)] = 20,
    measure_frames: Annotated[int, typer.Option("--frames", min=10)] = 300,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Jalankan seluruh evaluasi detektor, classifier, ablation end-to-end, & benchmark Bab 4."""
    console.print(f"[cyan]Menjalankan evaluasi lengkap Bab 4 (tag={tag}, split={split})...[/cyan]")
    summary = run_full_evaluation(
        tag=tag,
        detector_weights=detector_weights,
        classifier_weights=classifier_weights,
        split=split,
        warmup_frames=warmup_frames,
        measure_frames=measure_frames,
        output_dir=output_dir,
    )
    e2e_acc = summary["end_to_end"]["ablation_overall"]["fusion"]["accuracy"]
    console.print(
        f"[green]Laporan Bab 4 selesai di: {summary['output_dir']}[/green]\n"
        f"Detektor mAP@50: {summary['detector']['overall']['mAP50']:.4f} | "
        f"Classifier Macro-F1: {summary['classifier']['metrics']['macro_f1']:.4f} | "
        f"End-to-End Fusion Acc: {e2e_acc:.4f} | "
        f"FPS: {summary['benchmark']['fps_mean']}"
    )


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host", help="Host bind server.")] = None,
    port: Annotated[int | None, typer.Option("--port", min=1, max=65535)] = None,
    reload: Annotated[
        bool, typer.Option("--reload", help="Auto-reload untuk pengembangan.")
    ] = False,
) -> None:
    """Jalankan aplikasi demo web single-page FastAPI."""
    import torch
    import uvicorn

    app_cfg = load_app_config()
    bind_host = host or app_cfg.server.host
    bind_port = port or app_cfg.server.port
    if torch.cuda.is_available() and app_cfg.inference.device != "cpu":
        hw_info = f"GPU CUDA ({torch.cuda.get_device_name(0)})"
    else:
        hw_info = "CPU (CUDA tidak tersedia — fallback otomatis)"
    console.print(
        f"[green]Menjalankan aplikasi demo di http://{bind_host}:{bind_port}[/green] "
        f"[cyan][{hw_info}][/cyan]"
    )
    uvicorn.run(
        "attire_compliance.app.main:app",
        host=bind_host,
        port=bind_port,
        reload=reload,
    )
