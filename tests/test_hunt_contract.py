from pathlib import Path

from typer.testing import CliRunner

from veilscan import __version__, hunt_path
from veilscan.cli import app
from veilscan.decode.score import looks_like_token, score_text
from veilscan.hunt.gym import run_gym
from veilscan.hunt.types import HUNT_SCHEMA

runner = CliRunner()
TOKEN = "INV_WM:LEFT_EYE:2026"


def test_version_is_v2() -> None:
    assert __version__ == "2.0.0"


def test_hunt_help() -> None:
    r = runner.invoke(app, ["hunt", "--help"])
    assert r.exit_code == 0, r.output
    assert "--json" in r.output
    assert "--out" in r.output
    assert "--flag-re" in r.output


def test_short_photo_token_rejected() -> None:
    assert looks_like_token("PTQ:B7C") is False
    assert score_text("PTQ:B7C", framed=True) == 0.0
    assert looks_like_token(TOKEN) is True
    assert score_text(TOKEN, framed=True) >= 0.85


def test_gym_extracts_phase1_flags(tmp_path: Path) -> None:
    report = run_gym(tmp_path)
    assert report["n"] == 4
    assert report["hits"] == 4
    assert report["tpr"] == 1.0
    names = {row["name"] for row in report["cases"]}
    assert names == {"png-tEXt-flag", "jpeg-com-flag", "trailing-zip", "png-unknown-chunk"}


def test_hunt_cli_json_trailing_zip(tmp_path: Path) -> None:
    from veilscan.hunt.gym import write_gym

    cases = write_gym(tmp_path)
    zip_case = next(c for c in cases if c.name == "trailing-zip")
    out = tmp_path / "hunt-out"
    r = runner.invoke(app, ["hunt", str(zip_case.path), "--json", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert "FLAG{trailing-zip}" in r.stdout
    assert '"schema_version": 1' in r.stdout or f'"schema_version": {HUNT_SCHEMA}' in r.stdout
    assert (out / "findings.json").is_file()
    result = hunt_path(zip_case.path)
    assert zip_case.flag in result.flags


def test_gym_cli(tmp_path: Path) -> None:
    r = runner.invoke(app, ["gym", "--out", str(tmp_path / "g")])
    assert r.exit_code == 0, r.output
    assert "tpr=1.000" in r.output or "tpr=1.00" in r.output


def test_kitten_photo_does_not_decode_junk_token() -> None:
    photo = Path.home() / "Desktop" / "HR9f2gjWgAc2xxS.png"
    if not photo.is_file():
        return
    from veilscan.decode import decode_path

    dec = decode_path(photo)
    assert dec.text != "PTQ:B7C"
    if dec.found:
        assert dec.text and "PTQ:B7C" not in dec.text
    hunted = hunt_path(photo)
    assert not any("PTQ:B7C" in f for f in hunted.flags)
