from typer.testing import CliRunner

from veilscan.cli import app

runner = CliRunner()


def test_list_detectors_cli() -> None:
    r = runner.invoke(app, ["list-detectors"])
    assert r.exit_code == 0
    assert "chi_square" in r.stdout
    assert "fsnet_lite" in r.stdout


def test_version_cli() -> None:
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0
    assert "0.2.0" in r.stdout
