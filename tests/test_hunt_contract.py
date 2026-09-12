from pathlib import Path

import typer
from typer.testing import CliRunner

from veilscan import __version__, hunt_path
from veilscan.cli import app
from veilscan.decode.score import looks_like_token, score_text
from veilscan.hunt.gym import run_gym
from veilscan.hunt.types import HUNT_SCHEMA

runner = CliRunner()
TOKEN = "INV_WM:LEFT_EYE:2026"


def test_version_is_v2() -> None:
    assert __version__ == "2.2.0"


def test_hunt_help() -> None:
    # Rich help inserts ANSI between the two dashes in "--json"
    # ("-\x1b[0m\x1b[1m-json"), so substring checks on rendered help fail in CI.
    hunt_cmd = typer.main.get_command(app).commands["hunt"]
    opts = {flag for p in hunt_cmd.params for flag in p.opts}
    assert "--json" in opts
    assert "--out" in opts
    assert "--flag-re" in opts
    assert "--deep" in opts
    assert "--wordlist" in opts
    r = runner.invoke(app, ["hunt", "--help"], color=False)
    assert r.exit_code == 0, r.output


def test_short_photo_token_rejected() -> None:
    assert looks_like_token("PTQ:B7C") is False
    assert score_text("PTQ:B7C", framed=True) == 0.0
    assert looks_like_token(TOKEN) is True
    assert score_text(TOKEN, framed=True) >= 0.85


def test_gym_extracts_required_flags(tmp_path: Path) -> None:
    report = run_gym(tmp_path)
    assert report["tpr"] == 1.0
    assert report["required_miss"] == []
    names = {row["name"] for row in report["cases"] if not row.get("skipped")}
    core = {
        "png-tEXt-flag",
        "jpeg-com-flag",
        "trailing-zip",
        "png-unknown-chunk",
        "zsteg-rgb-bit0",
        "alpha-lsb",
        "palette-lsb",
        "bitplane-qr",
    }
    assert core <= names
    by = {row["name"]: row for row in report["cases"]}
    if not by["jsteg-flag"].get("skipped"):
        assert by["jsteg-flag"]["hit"] is True
    if not by["steghide-password"].get("skipped"):
        assert by["steghide-password"]["hit"] is True


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


def test_hunt_writes_bitplane_sheet(tmp_path: Path) -> None:
    from veilscan.hunt.gym import write_gym

    cases = write_gym(tmp_path)
    rgb_case = next(c for c in cases if c.name == "zsteg-rgb-bit0")
    out = tmp_path / "sheet-out"
    result = hunt_path(rgb_case.path, out_dir=out)
    assert rgb_case.flag in result.flags
    assert (out / "bitplanes.png").is_file()
    assert "bitplanes.png" in result.artifacts


def test_jsteg_roundtrip_when_jpeglib_present(tmp_path: Path) -> None:
    from veilscan.decode.jsteg import available
    from veilscan.hunt.gym import plant_jsteg, _cover

    if not available():
        return
    flag = "FLAG{jsteg-ac}"
    p = tmp_path / "jsteg-flag.jpg"
    p.write_bytes(plant_jsteg(_cover(38, 128), flag))
    result = hunt_path(p)
    assert flag in result.flags
    assert any(f.family == "jsteg" for f in result.findings)


def test_adapters_note_when_no_binaries(tmp_path: Path) -> None:
    from veilscan.hunt.adapters import which_tools

    tools = which_tools()
    if any(tools.values()):
        return
    from veilscan.decode.container import plant_jpeg_com
    from veilscan.hunt.gym import _cover

    p = tmp_path / "plain.jpg"
    p.write_bytes(plant_jpeg_com(_cover(32), "hello"))
    wl = tmp_path / "wl.txt"
    wl.write_text("veilscan-gym\n", encoding="utf-8")
    result = hunt_path(p, wordlist=wl)
    joined = " ".join(result.notes).lower()
    assert "adapter" in joined or "stegseek" in joined or "wordlist unused" in joined


def test_palette_survives_without_rgb_expand(tmp_path: Path) -> None:
    from PIL import Image

    from veilscan.hunt.gym import plant_palette_lsb

    flag = "FLAG{palette-lsb}"
    p = tmp_path / "p.png"
    p.write_bytes(plant_palette_lsb(flag, size=80, seed=41))
    assert Image.open(p).mode == "P"
    result = hunt_path(p)
    assert flag in result.flags
    assert any(f.family == "palette" for f in result.findings)


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
    assert hunted.flags == []
    assert not any(f.flag_hit for f in hunted.findings)
