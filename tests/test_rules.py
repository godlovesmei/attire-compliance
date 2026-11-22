"""Unit test untuk rule engine etika berpakaian (6 kelas terkunci + Opsi A)."""

from __future__ import annotations

from attire_compliance.pipeline.rules import evaluate_rules


def test_compliant_with_collared_shirt() -> None:
    res = evaluate_rules(["pakaian_berkerah", "bawahan_panjang", "sepatu"])
    assert res.status == "SESUAI"
    assert res.active_violations == ()
    assert res.missing_required == ()


def test_compliant_with_almamater_only() -> None:
    res = evaluate_rules(["jas_almamater", "bawahan_panjang", "sepatu"])
    assert res.status == "SESUAI"


def test_almamater_over_tshirt_is_compliant() -> None:
    """Keputusan terkunci: jas almamater di atas kaos oblong tetap SESUAI."""
    res = evaluate_rules(["jas_almamater", "kaos_oblong", "bawahan_panjang", "sepatu"])
    assert res.status == "SESUAI"
    assert res.active_violations == ()


def test_tshirt_without_almamater_is_violation() -> None:
    res = evaluate_rules(["kaos_oblong", "bawahan_panjang", "sepatu"])
    assert res.status == "TIDAK SESUAI"
    assert "kaos_oblong" in res.active_violations
    assert "Kaos Oblong" in res.reason


def test_sandal_is_always_violation_even_with_almamater() -> None:
    res = evaluate_rules(["jas_almamater", "bawahan_panjang", "sandal", "sepatu"])
    assert res.status == "TIDAK SESUAI"
    assert "sandal" in res.active_violations


def test_missing_required_when_body_clipped_returns_belum_lengkap() -> None:
    res = evaluate_rules(
        ["pakaian_berkerah", "bawahan_panjang"],
        full_body_visible=False,
        allow_incomplete=True,
    )
    assert res.status == "BELUM LENGKAP"
    assert "sepatu" in res.missing_required


def test_missing_required_when_full_body_visible_returns_tidak_sesuai() -> None:
    res = evaluate_rules(
        ["pakaian_berkerah", "bawahan_panjang"],
        full_body_visible=True,
        allow_incomplete=True,
    )
    assert res.status == "TIDAK SESUAI"
    assert "sepatu" in res.missing_required
