"""v2.4 signal case: WAV LSB, spectrogram stills, nested carve, folder case."""

import json
from pathlib import Path

from veilscan.decode.container import sniff_kind
from veilscan.hunt.audio import parse_wav, plant_wav_lsb, plant_wav_qr, plant_wav_tone
from veilscan.hunt.case import CASE_SCHEMA, hunt_case
from veilscan.hunt.gym import run_gym
from veilscan.hunt.pipeline import hunt_bytes, hunt_path


def test_flag_regex_drops_binary_tail() -> None:
    from veilscan.hunt.flags import find_flags

    junk = "FLAG{bmp-r-bit0} FLAG{bmp-r\x11nope} tail"
    assert find_flags(junk) == ["FLAG{bmp-r-bit0}"]


def test_sniff_wav_not_webp() -> None:
    data = plant_wav_lsb("FLAG{wav-pcm16-lsb}")
    assert sniff_kind(data) == "wav"
    assert parse_wav(data) is not None
    assert parse_wav(data).bits == 16


def test_wav_lsb_roundtrip(tmp_path: Path) -> None:
    flag = "FLAG{wav-pcm16-lsb}"
    path = tmp_path / "a.wav"
    path.write_bytes(plant_wav_lsb(flag))
    assert flag.encode() not in path.read_bytes()
    result = hunt_path(path, out_dir=tmp_path / "out")
    assert flag in result.flags
    assert any(f.method.startswith("pcm16-ch0-b0-fwd-msb") for f in result.findings)
    assert (tmp_path / "out" / "spectrogram.png").is_file()
    assert (tmp_path / "out" / "report.png").is_file()


def test_wav_stereo_right_and_reversed(tmp_path: Path) -> None:
    right = "FLAG{wav-stereo-right}"
    rev = "FLAG{wav-lsb-reversed}"
    rpath = tmp_path / "right.wav"
    vpath = tmp_path / "rev.wav"
    rpath.write_bytes(plant_wav_lsb(right, channels=2, channel=1))
    vpath.write_bytes(plant_wav_lsb(rev, reverse=True))
    assert right in hunt_path(rpath).flags
    assert rev in hunt_path(vpath).flags
    # Forward-only silence on the left channel must not be required to carry it.
    left_only = hunt_bytes(rpath.read_bytes())
    assert right in left_only.flags


def test_wav_tone_and_qr(tmp_path: Path) -> None:
    tone = "FLAG{wav-tone-bytes}"
    tpath = tmp_path / "tone.wav"
    tpath.write_bytes(plant_wav_tone(tone))
    assert tone.encode() not in tpath.read_bytes()
    assert tone in hunt_path(tpath).flags

    qr = "FLAG{wav-spectrogram-qr}"
    raw = plant_wav_qr(qr)
    assert raw is not None
    qpath = tmp_path / "qr.wav"
    qpath.write_bytes(raw)
    result = hunt_path(qpath, out_dir=tmp_path / "qr-out")
    assert qr in result.flags
    assert any(f.method == "spectrogram-qr" for f in result.findings)


def test_nested_zip_png_ztxt_not_plaintext(tmp_path: Path) -> None:
    from veilscan.hunt.gym import _cover, _plant_zip_png_ztxt

    flag = "FLAG{zip-png-ztxt}"
    raw = _plant_zip_png_ztxt(_cover(7), flag)
    assert flag.encode() not in raw
    path = tmp_path / "nest.png"
    path.write_bytes(raw)
    result = hunt_path(path)
    assert flag in result.flags
    assert any("zTXt" in f.method for f in result.findings)


def test_case_folder_unions_flags(tmp_path: Path) -> None:
    folder = tmp_path / "drop"
    folder.mkdir()
    (folder / "a.wav").write_bytes(plant_wav_lsb("FLAG{case-wav}"))
    from veilscan.generators import synthetic_cover
    from veilscan.decode.container import plant_png_text
    import numpy as np
    from PIL import Image
    import io

    png = plant_png_text(synthetic_cover(32, 32, np.random.default_rng(3), style="sine"), "FLAG{case-png}")
    (folder / "b.png").write_bytes(png)
    # noise file the case walker must ignore
    (folder / "notes.txt").write_text("FLAG{not-an-image}", encoding="utf-8")
    out = tmp_path / "case-out"
    report = hunt_case(folder, out_dir=out)
    assert report["schema_version"] == CASE_SCHEMA
    assert report["flag_count"] == 2
    assert "FLAG{case-wav}" in report["flags"]
    assert "FLAG{case-png}" in report["flags"]
    assert "FLAG{not-an-image}" not in report["flags"]
    js = json.loads((out / "case.json").read_text(encoding="utf-8"))
    assert js["flags"] == report["flags"]
    assert Image.open(io.BytesIO(png)).size[0] == 32


def test_gym_reaches_twenty(tmp_path: Path) -> None:
    report = run_gym(tmp_path)
    assert report["n"] >= 20
    assert report["tpr"] == 1.0
    assert report["required_miss"] == []
    names = {row["name"] for row in report["cases"] if row["required"] and not row.get("skipped")}
    assert {
        "gif-comment",
        "webp-xmp",
        "bmp-r-bit0",
        "zsteg-col-bit0",
        "wav-pcm16-lsb",
        "wav-pcm8-lsb",
        "wav-stereo-right",
        "wav-lsb-reversed",
        "wav-spectrogram-tone",
        "zip-png-ztxt",
    } <= names
    by = {row["name"]: row for row in report["cases"]}
    if not by["wav-spectrogram-qr"].get("skipped"):
        assert by["wav-spectrogram-qr"]["hit"] is True


def test_case_cli(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from veilscan.cli import app

    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "a.wav").write_bytes(plant_wav_lsb("FLAG{cli-case}"))
    out = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(app, ["case", str(folder), "--json", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "FLAG{cli-case}" in result.stdout
    assert (out / "case.json").is_file()
