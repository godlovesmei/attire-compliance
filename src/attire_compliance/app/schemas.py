"""Skema Pydantic v2 untuk REST API dan WebSocket aplikasi demo FastAPI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AttributeItemSchema(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class PersonResultSchema(BaseModel):
    track_id: int
    bbox: list[float]
    confidence: float
    full_body_visible: bool
    status: Literal["SESUAI", "TIDAK SESUAI", "BELUM LENGKAP"]
    raw_status: Literal["SESUAI", "TIDAK SESUAI", "BELUM LENGKAP"]
    reason: str
    prob_sesuai: float | None = None
    prob_tidak_sesuai: float | None = None
    active_violations: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    attributes: list[AttributeItemSchema] = Field(default_factory=list)


class TimingSchema(BaseModel):
    person_ms: float
    attribute_ms: float
    classifier_ms: float
    fusion_ms: float
    total_ms: float
    fps: float


class FrameAnalysisResponse(BaseModel):
    frame_width: int
    frame_height: int
    persons: list[PersonResultSchema]
    unassigned_attributes: list[AttributeItemSchema]
    timing: TimingSchema
    active_conf: float
    classifier_loaded: bool


class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    gpu_name: str | None
    person_detector: str
    attribute_detector: str | None
    classifier: str | None
    fusion_mode: str


class ClassMetaSchema(BaseModel):
    id: int
    name: str
    display_name: str
    role: str


class ConfigResponse(BaseModel):
    classes: list[ClassMetaSchema]
    default_conf: float
    iou_nms: float
    fusion_mode: str
    smoothing_window: int


class CaptureRequest(BaseModel):
    image_base64: str
    analysis: dict[str, object] = Field(default_factory=dict)


class SessionExportRequest(BaseModel):
    format: Literal["csv", "json"] = "csv"
    records: list[dict[str, object]] = Field(default_factory=list)
