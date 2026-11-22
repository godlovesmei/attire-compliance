"""Rule engine deterministik etika berpakaian mahasiswa Polibatam (6 kelas terkunci)."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

ComplianceStatus = Literal["SESUAI", "TIDAK SESUAI", "BELUM LENGKAP"]

DISPLAY_LABELS: dict[str, str] = {
    "pakaian_berkerah": "Pakaian Berkerah",
    "jas_almamater": "Jas Almamater",
    "kaos_oblong": "Kaos Oblong",
    "bawahan_panjang": "Bawahan Panjang",
    "sepatu": "Sepatu",
    "sandal": "Sandal",
}


@dataclass(frozen=True)
class RuleEvaluation:
    """Hasil evaluasi deterministik berdasarkan himpunan atribut yang terdeteksi."""

    status: ComplianceStatus
    reason: str
    active_violations: tuple[str, ...]
    missing_required: tuple[str, ...]
    top_ok: bool
    bottom_ok: bool
    foot_ok: bool


def evaluate_rules(
    detected_classes: Collection[str],
    *,
    full_body_visible: bool = True,
    almamater_overrides_tshirt: bool = True,
    allow_incomplete: bool = True,
) -> RuleEvaluation:
    """Evaluasi kepatuhan berdasarkan 6 kelas atribut terkunci (Opsi A: `bawahan_panjang`).

    Aturan terkunci:
    - `sandal` selalu menjadi pelanggaran.
    - `kaos_oblong` menjadi pelanggaran KECUALI tertutup `jas_almamater`
      (bila `almamater_overrides_tshirt=True`).
    - Atribut wajib: (`pakaian_berkerah` ATAU `jas_almamater`) DAN `bawahan_panjang` DAN `sepatu`.
    """
    attrs = set(detected_classes)

    violations: list[str] = []
    if "sandal" in attrs:
        violations.append("sandal")
    if "kaos_oblong" in attrs and not (almamater_overrides_tshirt and "jas_almamater" in attrs):
        violations.append("kaos_oblong")

    top_ok = ("pakaian_berkerah" in attrs) or ("jas_almamater" in attrs)
    bottom_ok = "bawahan_panjang" in attrs
    foot_ok = "sepatu" in attrs

    missing: list[str] = []
    if not top_ok:
        missing.append("atasan formal (pakaian berkerah / jas almamater)")
    if not bottom_ok:
        missing.append("bawahan panjang")
    if not foot_ok:
        missing.append("sepatu")

    if violations:
        names = ", ".join(DISPLAY_LABELS.get(v, v) for v in violations)
        return RuleEvaluation(
            status="TIDAK SESUAI",
            reason=f"Terdeteksi pelanggaran: {names}",
            active_violations=tuple(violations),
            missing_required=tuple(missing),
            top_ok=top_ok,
            bottom_ok=bottom_ok,
            foot_ok=foot_ok,
        )

    if top_ok and bottom_ok and foot_ok:
        return RuleEvaluation(
            status="SESUAI",
            reason="Seluruh atribut wajib terpenuhi",
            active_violations=(),
            missing_required=(),
            top_ok=True,
            bottom_ok=True,
            foot_ok=True,
        )

    missing_str = ", ".join(missing)
    if not full_body_visible and allow_incomplete:
        return RuleEvaluation(
            status="BELUM LENGKAP",
            reason=f"Tubuh belum terlihat utuh (belum terdeteksi: {missing_str})",
            active_violations=(),
            missing_required=tuple(missing),
            top_ok=top_ok,
            bottom_ok=bottom_ok,
            foot_ok=foot_ok,
        )

    return RuleEvaluation(
        status="TIDAK SESUAI",
        reason=f"Atribut wajib tidak terdeteksi: {missing_str}",
        active_violations=(),
        missing_required=tuple(missing),
        top_ok=top_ok,
        bottom_ok=bottom_ok,
        foot_ok=foot_ok,
    )
