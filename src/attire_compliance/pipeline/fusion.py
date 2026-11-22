"""Decision Fusion (Rule Engine + EfficientNetV2-S) beserta mode ablation Bab 4."""

from __future__ import annotations

from dataclasses import dataclass

from attire_compliance.config import FusionMode
from attire_compliance.pipeline.rules import ComplianceStatus, RuleEvaluation


@dataclass(frozen=True)
class FusionDecision:
    """Keputusan akhir kepatuhan per orang beserta transparansi alasan & probabilitas."""

    status: ComplianceStatus
    reason: str
    mode: FusionMode
    rule_status: ComplianceStatus
    prob_sesuai: float | None
    prob_tidak_sesuai: float | None
    active_violations: tuple[str, ...]
    missing_required: tuple[str, ...]


def fuse_decision(
    rule_eval: RuleEvaluation,
    *,
    prob_sesuai: float | None,
    mode: FusionMode = "fusion",
    classifier_veto_threshold: float = 0.80,
    full_body_visible: bool = True,
    allow_incomplete: bool = True,
) -> FusionDecision:
    """Gabungkan hasil rule engine YOLOv12 dengan probabilitas EfficientNetV2-S.

    Mendukung 3 mode untuk studi ablation di Bab 4:
    - `fusion`: aturan pelanggaran keras + veto visual EfficientNetV2 bila `p_tidak >= tau_veto`
    - `rule_only`: murni keputusan rule engine YOLOv12
    - `classifier_only`: murni keputusan probabilitas EfficientNetV2-S
    """
    prob_tidak = (1.0 - prob_sesuai) if prob_sesuai is not None else None

    if mode == "rule_only" or (mode == "fusion" and prob_sesuai is None):
        return FusionDecision(
            status=rule_eval.status,
            reason=rule_eval.reason,
            mode=mode,
            rule_status=rule_eval.status,
            prob_sesuai=prob_sesuai,
            prob_tidak_sesuai=prob_tidak,
            active_violations=rule_eval.active_violations,
            missing_required=rule_eval.missing_required,
        )

    if mode == "classifier_only":
        if not full_body_visible and allow_incomplete:
            return FusionDecision(
                status="BELUM LENGKAP",
                reason="Tubuh belum terlihat utuh untuk klasifikasi visual",
                mode=mode,
                rule_status=rule_eval.status,
                prob_sesuai=prob_sesuai,
                prob_tidak_sesuai=prob_tidak,
                active_violations=rule_eval.active_violations,
                missing_required=rule_eval.missing_required,
            )
        if prob_sesuai is None:
            return FusionDecision(
                status=rule_eval.status,
                reason="Classifier belum tersedia; fallback ke rule engine",
                mode=mode,
                rule_status=rule_eval.status,
                prob_sesuai=None,
                prob_tidak_sesuai=None,
                active_violations=rule_eval.active_violations,
                missing_required=rule_eval.missing_required,
            )
        assert prob_tidak is not None
        if prob_sesuai >= 0.5:
            return FusionDecision(
                status="SESUAI",
                reason=f"Klasifikasi visual EfficientNetV2: Sesuai ({prob_sesuai * 100:.1f}%)",
                mode=mode,
                rule_status=rule_eval.status,
                prob_sesuai=prob_sesuai,
                prob_tidak_sesuai=prob_tidak,
                active_violations=rule_eval.active_violations,
                missing_required=rule_eval.missing_required,
            )
        return FusionDecision(
            status="TIDAK SESUAI",
            reason=f"Klasifikasi visual EfficientNetV2: Tidak Sesuai ({prob_tidak * 100:.1f}%)",
            mode=mode,
            rule_status=rule_eval.status,
            prob_sesuai=prob_sesuai,
            prob_tidak_sesuai=prob_tidak,
            active_violations=rule_eval.active_violations,
            missing_required=rule_eval.missing_required,
        )

    # Mode default: "fusion"
    assert prob_tidak is not None
    if rule_eval.active_violations:
        return FusionDecision(
            status="TIDAK SESUAI",
            reason=rule_eval.reason,
            mode=mode,
            rule_status=rule_eval.status,
            prob_sesuai=prob_sesuai,
            prob_tidak_sesuai=prob_tidak,
            active_violations=rule_eval.active_violations,
            missing_required=rule_eval.missing_required,
        )

    if rule_eval.top_ok and rule_eval.bottom_ok and rule_eval.foot_ok:
        if prob_tidak >= classifier_veto_threshold:
            return FusionDecision(
                status="TIDAK SESUAI",
                reason=(
                    f"Ditolak klasifikasi visual EfficientNetV2 "
                    f"(p_tidak = {prob_tidak * 100:.1f}%)"
                ),
                mode=mode,
                rule_status=rule_eval.status,
                prob_sesuai=prob_sesuai,
                prob_tidak_sesuai=prob_tidak,
                active_violations=(),
                missing_required=(),
            )
        return FusionDecision(
            status="SESUAI",
            reason="Sesuai etika berpakaian (YOLOv12 + EfficientNetV2)",
            mode=mode,
            rule_status=rule_eval.status,
            prob_sesuai=prob_sesuai,
            prob_tidak_sesuai=prob_tidak,
            active_violations=(),
            missing_required=(),
        )

    return FusionDecision(
        status=rule_eval.status,
        reason=rule_eval.reason,
        mode=mode,
        rule_status=rule_eval.status,
        prob_sesuai=prob_sesuai,
        prob_tidak_sesuai=prob_tidak,
        active_violations=rule_eval.active_violations,
        missing_required=rule_eval.missing_required,
    )
