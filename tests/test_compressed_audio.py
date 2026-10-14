"""v2.6 MP3/FLAC metadata, verbatim FLAC LSB, spectrogram-letter OCR."""

import shutil

from typer.testing import CliRunner

from veilscan.cli import app
from veilscan.decode.container import sniff_kind
from veilscan.hunt.audio import plant_wav_glyphs
from veilscan.hunt.case import hunt_case
from veilscan.hunt.compressed_audio import plant_flac_comment, plant_flac_lsb, plant_mp3_id3
from veilscan.hunt.gym import write_gym
from veilscan.hunt.pipeline import hunt_bytes

runner = CliRunner()


def test_id3_utf16_is_not_plaintext_and_hunt_reads_it() -> None:
    flag = "FLAG{mp3-id3}"
    blob = plant_mp3_id3(flag)
    assert sniff_kind(blob) == "mp3"
    assert b"FLAG{" not in blob
    result = hunt_bytes(blob, source="id3.mp3")
    assert flag in result.flags
    assert any(f.method == "id3" for f in result.findings)


def test_flac_comment_and_verbatim_lsb(tmp_path) -> None:
    comment = plant_flac_comment("FLAG{flac-comment}")
    assert sniff_kind(comment) == "flac"
    commented = hunt_bytes(comment, source="c.flac")
    assert "FLAG{flac-comment}" in commented.flags
    assert any(f.method == "flac-comment" for f in commented.findings)

    flag = "FLAG{flac-lsb}"
    blob = plant_flac_lsb(flag)
    assert b"FLAG{" not in blob
    result = hunt_bytes(blob, source="lsb.flac")
    assert flag in result.flags
    assert any(str(f.method).startswith("pcm16") for f in result.findings)


def test_scan_refuses_audio_containers(tmp_path) -> None:
    mp3 = tmp_path / "a.mp3"
    flac = tmp_path / "a.flac"
    mp3.write_bytes(plant_mp3_id3("FLAG{mp3-id3}"))
    flac.write_bytes(plant_flac_comment("FLAG{flac-comment}"))
    for path, label in ((mp3, "mp3"), (flac, "flac")):
        scan = runner.invoke(app, ["scan", str(path)])
        assert scan.exit_code != 0
        assert "hunt" in scan.output.lower()
        assert label.upper() in scan.output or label in scan.output.lower()


def test_glyph_wav_has_no_plaintext_flag() -> None:
    flag = "FLAG{SPEC-INK}"
    blob = plant_wav_glyphs(flag)
    assert b"FLAG{" not in blob
    assert b"SPEC-INK" not in blob
    result = hunt_bytes(blob, source="ink.wav")
    if shutil.which("tesseract"):
        assert flag in result.flags
        assert any(f.method == "spectrogram-ocr" for f in result.findings)
    else:
        assert flag not in result.flags


def test_case_reads_mp3_and_flac(tmp_path) -> None:
    folder = tmp_path / "case"
    folder.mkdir()
    (folder / "note.mp3").write_bytes(plant_mp3_id3("FLAG{case-mp3}"))
    (folder / "note.flac").write_bytes(plant_flac_comment("FLAG{case-flac}"))
    (folder / "notes.txt").write_text("FLAG{not-audio}", encoding="utf-8")
    report = hunt_case(folder)
    assert report["flags"] == ["FLAG{case-flac}", "FLAG{case-mp3}"] or set(report["flags"]) == {
        "FLAG{case-flac}",
        "FLAG{case-mp3}",
    }
    assert "FLAG{not-audio}" not in report["flags"]


def test_gym_lists_audio_rows(tmp_path) -> None:
    names = {case.name for case in write_gym(tmp_path)}
    assert {"mp3-id3", "flac-comment", "flac-lsb", "wav-spectrogram-ocr"} <= names
