"""Unit test untuk Decision Fusion & 3 mode ablation (fusion, rule_only, classifier_only)."""

from __future__ import annotations

from attire_compliance.pipeline.fusion import fuse_decision
from attire_compliance.pipeline.rules import evaluate_rules


def test_fusion_accepts_compliant_when_classifier_agrees() -> None:
    rule_eval = evaluate_rules(["pakaian_berkerah", "bawahan_panjang", "sepatu"])
    decision = fuse_decision(rule_eval, prob_sesuai=0.92, mode="fusion")
    assert decision.status == "SESUAI"
    assert decision.prob_sesuai == 0.92


def test_fusion_vetoes_when_classifier_strongly_rejects() -> None:
    rule_eval = evaluate_rules(["pakaian_berkerah", "bawahan_panjang", "sepatu"])
    decision = fuse_decision(
        rule_eval, prob_sesuai=0.15, mode="fusion", classifier_veto_threshold=0.80
    )
    assert decision.status == "TIDAK SESUAI"
    assert "EfficientNetV2" in decision.reason


def test_fusion_hard_violation_overrides_high_classifier_prob() -> None:
    rule_eval = evaluate_rules(["pakaian_berkerah", "bawahan_panjang", "sandal"])
    decision = fuse_decision(rule_eval, prob_sesuai=0.99, mode="fusion")
    assert decision.status == "TIDAK SESUAI"
    assert "sandal" in decision.active_violations


def test_fusion_fallback_when_classifier_missing() -> None:
    rule_eval = evaluate_rules(["jas_almamater", "bawahan_panjang", "sepatu"])
    decision = fuse_decision(rule_eval, prob_sesuai=None, mode="fusion")
    assert decision.status == "SESUAI"


def test_ablation_modes_rule_only_and_classifier_only() -> None:
    rule_eval = evaluate_rules(["pakaian_berkerah", "bawahan_panjang", "sepatu"])
    rule_only = fuse_decision(rule_eval, prob_sesuai=0.05, mode="rule_only")
    assert rule_only.status == "SESUAI"

    clf_only_bad = fuse_decision(rule_eval, prob_sesuai=0.20, mode="classifier_only")
    assert clf_only_bad.status == "TIDAK SESUAI"

    clf_only_incomplete = fuse_decision(
        rule_eval,
        prob_sesuai=0.90,
        mode="classifier_only",
        full_body_visible=False,
        allow_incomplete=True,
    )
    assert clf_only_incomplete.status == "BELUM LENGKAP"
