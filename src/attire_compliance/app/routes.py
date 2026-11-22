"""REST & WebSocket routes untuk aplikasi demo single-page attire-compliance."""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, Response

from attire_compliance.app.schemas import (
    AttributeItemSchema,
    CaptureRequest,
    ClassMetaSchema,
    ConfigResponse,
    FrameAnalysisResponse,
    HealthResponse,
    PersonResultSchema,
    SessionExportRequest,
    TimingSchema,
)
from attire_compliance.paths import resolve_project_path
from attire_compliance.pipeline.pipeline import AttireCompliancePipeline, FrameResult

router = APIRouter()


def _decode_image_bytes(raw: bytes) -> np.ndarray:
    buf = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Data gambar tidak dapat di-decode oleh OpenCV.")
    return frame


def _serialize_frame_result(
    result: FrameResult,
    *,
    active_conf: float,
    classifier_loaded: bool,
) -> FrameAnalysisResponse:
    persons = [
        PersonResultSchema(
            track_id=p.track_id,
            bbox=[round(v, 1) for v in p.bbox],
            confidence=round(p.confidence, 4),
            full_body_visible=p.full_body_visible,
            status=p.smoothed_status,
            raw_status=p.raw_decision.status,
            reason=p.raw_decision.reason,
            prob_sesuai=(
                round(p.raw_decision.prob_sesuai, 4)
                if p.raw_decision.prob_sesuai is not None
                else None
            ),
            prob_tidak_sesuai=(
                round(p.raw_decision.prob_tidak_sesuai, 4)
                if p.raw_decision.prob_tidak_sesuai is not None
                else None
            ),
            active_violations=list(p.raw_decision.active_violations),
            missing_required=list(p.raw_decision.missing_required),
            attributes=[
                AttributeItemSchema(
                    class_id=a.class_id,
                    class_name=a.class_name,
                    confidence=round(a.confidence, 4),
                    bbox=[round(v, 1) for v in a.bbox],
                )
                for a in p.attributes
            ],
        )
        for p in result.persons
    ]
    unassigned = [
        AttributeItemSchema(
            class_id=a.class_id,
            class_name=a.class_name,
            confidence=round(a.confidence, 4),
            bbox=[round(v, 1) for v in a.bbox],
        )
        for a in result.unassigned_attributes
    ]
    return FrameAnalysisResponse(
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        persons=persons,
        unassigned_attributes=unassigned,
        timing=TimingSchema(
            person_ms=result.timing.person_ms,
            attribute_ms=result.timing.attribute_ms,
            classifier_ms=result.timing.classifier_ms,
            fusion_ms=result.timing.fusion_ms,
            total_ms=result.timing.total_ms,
            fps=result.timing.fps,
        ),
        active_conf=active_conf,
        classifier_loaded=classifier_loaded,
    )


def _get_pipeline(app_state: Any) -> AttireCompliancePipeline:
    pipeline = getattr(app_state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline model belum siap.")
    return pipeline


def _get_executor(app_state: Any) -> ThreadPoolExecutor:
    return app_state.executor


@router.get("/", response_class=HTMLResponse)
async def index_page(request: Request) -> HTMLResponse:
    template_path: Path = request.app.state.template_path
    return HTMLResponse(template_path.read_text(encoding="utf-8"))


@router.get("/api/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    pipeline = _get_pipeline(request.app.state)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return HealthResponse(
        status="ok",
        cuda_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        person_detector=pipeline.person_weights_path.name,
        attribute_detector=(
            pipeline.attribute_weights_path.name
            if pipeline.attribute_weights_path is not None
            else None
        ),
        classifier=(
            pipeline.classifier_weights_path.name
            if pipeline.classifier_weights_path is not None
            else None
        ),
        fusion_mode=pipeline.app_cfg.fusion.mode,
    )


@router.get("/api/config", response_model=ConfigResponse)
async def get_config(request: Request) -> ConfigResponse:
    pipeline = _get_pipeline(request.app.state)
    return ConfigResponse(
        classes=[
            ClassMetaSchema(
                id=c.id,
                name=c.name,
                display_name=c.display_name,
                role=c.role,
            )
            for c in pipeline.classes_cfg.classes
        ],
        default_conf=pipeline.app_cfg.thresholds.attribute_conf_default,
        iou_nms=pipeline.app_cfg.thresholds.iou_nms,
        fusion_mode=pipeline.app_cfg.fusion.mode,
        smoothing_window=pipeline.app_cfg.fusion.smoothing_window,
    )


@router.post("/api/infer-image", response_model=FrameAnalysisResponse)
async def infer_image(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    conf: float | None = None,
) -> FrameAnalysisResponse:
    pipeline = _get_pipeline(request.app.state)
    executor = _get_executor(request.app.state)
    raw = await file.read()
    try:
        frame = _decode_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    active_conf = conf if conf is not None else pipeline.app_cfg.thresholds.attribute_conf_default
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: pipeline.process_frame(
            frame,
            conf_override=conf,
            track=False,
            use_smoothing=False,
        ),
    )
    return _serialize_frame_result(
        result,
        active_conf=active_conf,
        classifier_loaded=pipeline.classifier is not None,
    )


@router.post("/api/reset")
async def reset_session(request: Request) -> dict[str, str]:
    pipeline = _get_pipeline(request.app.state)
    executor = _get_executor(request.app.state)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, pipeline.reset_tracking)
    return {"status": "reset"}


@router.post("/api/capture")
async def capture_frame(request: Request, payload: CaptureRequest) -> dict[str, str]:
    pipeline = _get_pipeline(request.app.state)
    sessions_root = resolve_project_path(pipeline.app_cfg.server.sessions_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    capture_dir = sessions_root / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)

    b64_data = payload.image_base64.split(",", 1)[-1]
    try:
        img_bytes = base64.b64decode(b64_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Base64 gambar tidak valid: {exc}") from exc

    img_path = capture_dir / f"capture_{stamp}.jpg"
    meta_path = capture_dir / f"capture_{stamp}.json"
    img_path.write_bytes(img_bytes)
    meta_path.write_text(
        json.dumps(payload.analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"saved_image": str(img_path), "saved_json": str(meta_path)}


@router.post("/api/sessions/export")
async def export_session(request: Request, payload: SessionExportRequest) -> Response:
    pipeline = _get_pipeline(request.app.state)
    sessions_root = resolve_project_path(pipeline.app_cfg.server.sessions_dir)
    export_dir = sessions_root / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if payload.format == "json":
        content = json.dumps(payload.records, indent=2, ensure_ascii=False) + "\n"
        (export_dir / f"session_{stamp}.json").write_text(content, encoding="utf-8")
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="attire_session_{stamp}.json"'},
        )

    buf = io.StringIO()
    fieldnames = [
        "timestamp",
        "track_id",
        "status",
        "reason",
        "prob_sesuai",
        "attributes",
        "total_ms",
        "fps",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in payload.records:
        writer.writerow(row)
    csv_text = buf.getvalue()
    (export_dir / f"session_{stamp}.csv").write_text(csv_text, encoding="utf-8")
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="attire_session_{stamp}.csv"'},
    )


@router.websocket("/ws/infer")
async def ws_infer(websocket: WebSocket) -> None:
    """WebSocket inferensi real-time: terima frame JPEG biner atau pesan kontrol JSON."""
    await websocket.accept()
    pipeline = _get_pipeline(websocket.app.state)
    executor = _get_executor(websocket.app.state)
    loop = asyncio.get_running_loop()

    conf_override: float | None = None
    track_mode = True

    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                try:
                    cmd = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "detail": "JSON tidak valid"})
                    continue
                cmd_type = cmd.get("type")
                if cmd_type == "config":
                    if "conf" in cmd and cmd["conf"] is not None:
                        conf_override = max(0.05, min(0.95, float(cmd["conf"])))
                    if "track" in cmd:
                        track_mode = bool(cmd["track"])
                    await websocket.send_json(
                        {"type": "config_ack", "conf": conf_override, "track": track_mode}
                    )
                elif cmd_type == "reset":
                    await loop.run_in_executor(executor, pipeline.reset_tracking)
                    await websocket.send_json({"type": "reset_ack"})
                continue

            raw_bytes = msg.get("bytes")
            if not raw_bytes:
                continue

            try:
                frame = _decode_image_bytes(raw_bytes)
            except ValueError as exc:
                await websocket.send_json({"type": "error", "detail": str(exc)})
                continue

            active_conf = (
                conf_override
                if conf_override is not None
                else pipeline.app_cfg.thresholds.attribute_conf_default
            )
            result = await loop.run_in_executor(
                executor,
                lambda f=frame, c=conf_override, t=track_mode: pipeline.process_frame(
                    f,
                    conf_override=c,
                    track=t,
                    use_smoothing=t,
                ),
            )
            payload = _serialize_frame_result(
                result,
                active_conf=active_conf,
                classifier_loaded=pipeline.classifier is not None,
            )
            await websocket.send_json(payload.model_dump())
    except WebSocketDisconnect:
        pass
