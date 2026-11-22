"""Unit test untuk modul diagnostik & CLI `attire doctor`."""

from __future__ import annotations

from typer.testing import CliRunner

from attire_compliance.cli import app
from attire_compliance.doctor import check_configs, check_python_version, run_doctor

runner = CliRunner()


def test_check_configs_passes() -> None:
    result = check_configs()
    assert result.ok is True
    assert "bawahan_panjang" in result.detail


def test_check_python_version_matches_supported_range() -> None:
    result = check_python_version()
    assert result.ok is True


def test_run_doctor_skip_inference_succeeds() -> None:
    results = run_doctor(skip_inference=True)
    names = [item.name for item in results]
    assert names == ["Python", "Konfigurasi YAML", "Dependensi inti", "CUDA & GPU"]
    assert all(item.ok for item in results), [r for r in results if not r.ok]


def test_cli_version_command() -> None:
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "attire-compliance" in res.stdout


def test_cli_doctor_skip_inference() -> None:
    res = runner.invoke(app, ["doctor", "--skip-inference"])
    assert res.exit_code == 0, res.stdout
