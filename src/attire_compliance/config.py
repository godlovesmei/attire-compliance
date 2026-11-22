"""Loader dan validasi konfigurasi YAML berbasis Pydantic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from attire_compliance.paths import CONFIGS_DIR

ClassRole = Literal["required_top", "required_bottom", "required_foot", "violation"]
FusionMode = Literal["fusion", "rule_only", "classifier_only"]


class ClassItem(BaseModel):
    id: int = Field(ge=0)
    name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    role: ClassRole


class ClassesConfig(BaseModel):
    classes: list[ClassItem]

    @model_validator(mode="after")
    def validate_locked_classes(self) -> ClassesConfig:
        if len(self.classes) != 6:
            raise ValueError(f"Jumlah kelas harus tepat 6, ditemukan {len(self.classes)}.")
        ids = [item.id for item in self.classes]
        if ids != list(range(6)):
            raise ValueError(f"ID kelas harus berurutan 0..5, ditemukan {ids}.")
        names = [item.name for item in self.classes]
        if len(set(names)) != len(names):
            raise ValueError("Nama kelas tidak boleh duplikat.")
        return self

    @property
    def id_to_name(self) -> dict[int, str]:
        return {item.id: item.name for item in self.classes}

    @property
    def name_to_id(self) -> dict[str, int]:
        return {item.name: item.id for item in self.classes}

    @property
    def names(self) -> list[str]:
        return [item.name for item in self.classes]


class FashionpediaRule(BaseModel):
    source_category: str = Field(min_length=1)
    require_parts: list[str] = Field(default_factory=list)
    exclude_parts: list[str] = Field(default_factory=list)
    long_only: bool = False
    target_class: str = Field(min_length=1)


class FashionpediaSourceConfig(BaseModel):
    hf_dataset: str
    output_dir: str
    part_overlap_min_ioa: float = Field(gt=0.0, le=1.0)
    long_skirt_min_height_ratio: float = Field(gt=0.0, le=1.0)
    rules: list[FashionpediaRule]


class OpenImagesSourceConfig(BaseModel):
    zoo_name: str
    output_dir: str
    max_samples_per_split: dict[str, int]
    class_map: dict[str, str]


class RoboflowDatasetEntry(BaseModel):
    class_map: dict[str, str]


class RoboflowSourceConfig(BaseModel):
    root_dir: str
    datasets: dict[str, RoboflowDatasetEntry] = Field(default_factory=dict)


class SourcesConfig(BaseModel):
    fashionpedia: FashionpediaSourceConfig
    openimages: OpenImagesSourceConfig
    roboflow: RoboflowSourceConfig


class DetectorAugmentationConfig(BaseModel):
    hsv_h: float = Field(ge=0.0, le=1.0)
    hsv_s: float = Field(ge=0.0, le=1.0)
    hsv_v: float = Field(ge=0.0, le=1.0)
    degrees: float = Field(ge=0.0, le=180.0)
    translate: float = Field(ge=0.0, le=1.0)
    scale: float = Field(ge=0.0, le=1.0)
    fliplr: float = Field(ge=0.0, le=1.0)
    flipud: float = Field(ge=0.0, le=1.0)
    mosaic: float = Field(ge=0.0, le=1.0)
    mixup: float = Field(ge=0.0, le=1.0)


class DetectorConfig(BaseModel):
    model: str
    pretrained_dir: str
    data_yaml: str
    project: str
    run_name: str
    imgsz: int = Field(gt=0)
    batch: int = Field(gt=0)
    workers: int = Field(ge=0)
    epochs: int = Field(gt=0)
    patience: int = Field(ge=0)
    optimizer: str
    lr0: float = Field(gt=0.0)
    lrf: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    warmup_epochs: float = Field(ge=0.0)
    amp: bool
    cache: bool
    seed: int
    deterministic: bool
    device: str
    augmentation: DetectorAugmentationConfig


class ClassifierConfig(BaseModel):
    backbone: str
    pretrained: bool
    data_dir: str
    output_dir: str
    checkpoint_path: str
    class_names: list[str]
    imgsz: int = Field(gt=0)
    pad_ratio: float = Field(ge=0.0, lt=1.0)
    letterbox: bool
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    epochs: int = Field(gt=0)
    patience: int = Field(ge=0)
    optimizer: str
    lr: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    warmup_epochs: int = Field(ge=0)
    label_smoothing: float = Field(ge=0.0, lt=1.0)
    amp: bool
    seed: int
    device: str

    @field_validator("class_names")
    @classmethod
    def validate_binary_classes(cls, value: list[str]) -> list[str]:
        if len(value) != 2 or set(value) != {"tidak_sesuai", "sesuai"}:
            raise ValueError("class_names classifier harus terdiri dari tidak_sesuai dan sesuai.")
        return value


class ModelPathsConfig(BaseModel):
    person_detector: str
    attribute_detector: str
    classifier: str


class InferenceConfig(BaseModel):
    device: str
    half: bool
    imgsz_detector: int = Field(gt=0)
    imgsz_classifier: int = Field(gt=0)
    person_pad_ratio: float = Field(ge=0.0, lt=1.0)
    tracker: str


class ThresholdsConfig(BaseModel):
    person_conf: float = Field(gt=0.0, le=1.0)
    attribute_conf_default: float = Field(gt=0.0, le=1.0)
    iou_nms: float = Field(gt=0.0, le=1.0)
    association_min_ioa: float = Field(gt=0.0, le=1.0)
    per_class: dict[str, float] = Field(default_factory=dict)

    @field_validator("per_class")
    @classmethod
    def validate_threshold_values(cls, value: dict[str, float]) -> dict[str, float]:
        for name, score in value.items():
            if not (0.0 < score <= 1.0):
                raise ValueError(f"Threshold per_class[{name}] harus di rentang (0, 1].")
        return value


class FusionConfig(BaseModel):
    mode: FusionMode
    classifier_veto_threshold: float = Field(gt=0.0, le=1.0)
    almamater_overrides_tshirt: bool
    smoothing_window: int = Field(ge=1)
    bottom_edge_margin_px: int = Field(ge=0)


class ServerConfig(BaseModel):
    host: str
    port: int = Field(ge=1, le=65535)
    sessions_dir: str


class AppConfig(BaseModel):
    models: ModelPathsConfig
    inference: InferenceConfig
    thresholds: ThresholdsConfig
    fusion: FusionConfig
    server: ServerConfig


@dataclass(frozen=True)
class ProjectConfigs:
    classes: ClassesConfig
    sources: SourcesConfig
    detector: DetectorConfig
    classifier: ClassifierConfig
    app: AppConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Format YAML tidak valid (harus mapping) pada: {path}")
    return data


def _resolve_configs_dir(configs_dir: Path | None) -> Path:
    return configs_dir if configs_dir is not None else CONFIGS_DIR


def load_classes_config(configs_dir: Path | None = None) -> ClassesConfig:
    path = _resolve_configs_dir(configs_dir) / "classes.yaml"
    return ClassesConfig.model_validate(_read_yaml(path))


def load_sources_config(configs_dir: Path | None = None) -> SourcesConfig:
    path = _resolve_configs_dir(configs_dir) / "sources.yaml"
    return SourcesConfig.model_validate(_read_yaml(path))


def load_detector_config(configs_dir: Path | None = None) -> DetectorConfig:
    path = _resolve_configs_dir(configs_dir) / "detector.yaml"
    return DetectorConfig.model_validate(_read_yaml(path))


def load_classifier_config(configs_dir: Path | None = None) -> ClassifierConfig:
    return ClassifierConfig.model_validate(
        _read_yaml(_resolve_configs_dir(configs_dir) / "classifier.yaml")
    )


def load_app_config(configs_dir: Path | None = None) -> AppConfig:
    return AppConfig.model_validate(_read_yaml(_resolve_configs_dir(configs_dir) / "app.yaml"))


def load_all_configs(configs_dir: Path | None = None) -> ProjectConfigs:
    """Muat dan validasi lintas-file seluruh konfigurasi proyek."""
    classes = load_classes_config(configs_dir)
    sources = load_sources_config(configs_dir)
    detector = load_detector_config(configs_dir)
    classifier = load_classifier_config(configs_dir)
    app = load_app_config(configs_dir)

    valid_names = set(classes.names)

    for rule in sources.fashionpedia.rules:
        if rule.target_class not in valid_names:
            raise ValueError(
                f"sources.yaml fashionpedia target_class tidak valid: {rule.target_class}"
            )
    for src, target in sources.openimages.class_map.items():
        if target not in valid_names:
            raise ValueError(
                f"sources.yaml openimages class_map[{src}] tidak valid: {target}"
            )
    for ds_name, entry in sources.roboflow.datasets.items():
        for src, target in entry.class_map.items():
            if target not in valid_names:
                raise ValueError(
                    f"sources.yaml roboflow[{ds_name}].class_map[{src}] tidak valid: {target}"
                )
    for cls_name in app.thresholds.per_class:
        if cls_name not in valid_names:
            raise ValueError(f"app.yaml thresholds.per_class tidak valid: {cls_name}")

    return ProjectConfigs(
        classes=classes,
        sources=sources,
        detector=detector,
        classifier=classifier,
        app=app,
    )
