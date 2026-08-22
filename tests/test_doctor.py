from typer.testing import CliRunner

from veilscan.cli import app
from veilscan.doctor import run_doctor

runner = CliRunner()


def test_doctor_ok_json() -> None:
    report = run_doctor()
    assert "version" in report
    assert "checkpoints" in report
    assert report["operating_point"] is not None
    assert report["camera_operating_point"] is not None
    assert str(report["camera_operating_point"].get("id") or "").startswith("op-v1.0.0-camera")
    div = report.get("camera_div2k_operating_point") or {}
    assert str(div.get("id") or "").startswith("op-v1.0.0-camera-div2k")
    names = {row["name"] for row in report["checkpoints"]}
    assert "residual_cnn" in names
    assert "fsnet_lite" in names


def test_doctor_cli_exit_0() -> None:
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "operating_point" in r.stdout
    assert "residual_cnn" in r.stdout
