"""PDF hunt. Flags that are only inside streams must not be plaintext."""

from pathlib import Path

from typer.testing import CliRunner

from veilscan.cli import app
from veilscan.hunt.gym import write_gym
from veilscan.hunt.pdf import (
    plant_pdf_comment,
    plant_pdf_embed,
    plant_pdf_incremental,
    plant_pdf_info,
    plant_pdf_invisible,
    plant_pdf_js,
    plant_pdf_nested_png,
    plant_pdf_trail,
)
from veilscan.hunt.pipeline import hunt_bytes
from veilscan.decode.container import plant_png_ztxt
from veilscan.generators import synthetic_cover
import numpy as np

runner = CliRunner()


def _flags(data: bytes) -> tuple[list[str], list[str]]:
    result = hunt_bytes(data, source="mem.pdf")
    methods = [f"{f.family}/{f.method}" for f in result.findings]
    return result.flags, methods


def test_pdf_channels_survive_compression() -> None:
    hidden = {
        "js": plant_pdf_js("FLAG{pdf-js}"),
        "embed": plant_pdf_embed("FLAG{pdf-embed}"),
        "invisible": plant_pdf_invisible("FLAG{pdf-invisible}"),
    }
    for name, blob in hidden.items():
        assert b"FLAG{" not in blob, name
        flags, methods = _flags(blob)
        assert any(name in flag for flag in flags), (name, flags, methods)

    info = plant_pdf_info("FLAG{pdf-info}")
    flags, methods = _flags(info)
    assert "FLAG{pdf-info}" in flags
    assert "pdf/info" in methods
    assert "carve/pdf" not in methods

    trail = plant_pdf_trail("FLAG{pdf-trail}")
    flags, _methods = _flags(trail)
    assert "FLAG{pdf-trail}" in flags

    comment = plant_pdf_comment("FLAG{pdf-comment}")
    flags, methods = _flags(comment)
    assert "FLAG{pdf-comment}" in flags
    assert "pdf/comment" in methods

    old = plant_pdf_incremental("FLAG{pdf-old}")
    flags, methods = _flags(old)
    assert "FLAG{pdf-old}" in flags
    assert "pdf/superseded" in methods

    png = plant_png_ztxt(
        synthetic_cover(32, 32, np.random.default_rng(7), style="sine"),
        "FLAG{pdf-nested-png}",
        key="Comment",
    )
    nested = plant_pdf_nested_png(png)
    assert b"FLAG{pdf-nested-png}" not in nested
    flags, _methods = _flags(nested)
    assert "FLAG{pdf-nested-png}" in flags


def test_clean_pdf_has_no_flag_and_scan_refuses(tmp_path: Path) -> None:
    blob = plant_pdf_info("hello")
    flags, methods = _flags(blob)
    assert flags == []
    assert "pdf/structure" in methods
    assert "carve/pdf" not in methods
    path = tmp_path / "clean.pdf"
    path.write_bytes(blob)
    scan = runner.invoke(app, ["scan", str(path)])
    assert scan.exit_code != 0
    assert "hunt" in scan.output.lower()


def test_case_reads_pdf_and_skips_text(tmp_path: Path) -> None:
    from veilscan.hunt.case import hunt_case

    folder = tmp_path / "case"
    folder.mkdir()
    (folder / "note.pdf").write_bytes(plant_pdf_invisible("FLAG{case-pdf}"))
    (folder / "notes.txt").write_text("FLAG{not-a-pdf}", encoding="utf-8")
    report = hunt_case(folder)
    assert report["flag_count"] == 1
    assert report["flags"] == ["FLAG{case-pdf}"]


def test_gym_lists_pdf_rows(tmp_path: Path) -> None:
    names = {case.name for case in write_gym(tmp_path)}
    assert {
        "pdf-info",
        "pdf-js",
        "pdf-embed",
        "pdf-trail",
        "pdf-comment",
        "pdf-invisible",
        "pdf-incremental",
        "pdf-nested-png",
    } <= names
