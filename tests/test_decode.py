from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from veilscan.cli import app
from veilscan.decode import decode_bytes, decode_path
from veilscan.decode.report import default_report_path, render_decode_report
from veilscan.decode.container import plant_jpeg_com, plant_png_text
from veilscan.decode.embed_text import (
    embed_text_array,
    embed_text_patch,
    embed_text_path,
    embed_text_png_bytes,
)
from veilscan.decode.score import looks_like_token, score_text
from veilscan.decode.layouts import LSB_LAYOUTS
from veilscan.decode.lsb import extract_bits, pack_bits
from veilscan.decode.score import split_frames, utf8_text
from veilscan.generators import synthetic_cover

MSG = "hello veil"
runner = CliRunner()


def _cover(seed: int = 11, size: int = 64) -> np.ndarray:
    return synthetic_cover(size, size, np.random.default_rng(seed), style="sine")


def test_lsb_each_layout_roundtrip() -> None:
    cover = _cover()
    for row in LSB_LAYOUTS:
        marked = embed_text_array(cover, MSG, layout_id=row["id"])
        raw = pack_bits(extract_bits(marked, row), msb_first=bool(row["msb_first"]))
        texts = [utf8_text(payload) for payload, _framed in split_frames(raw)]
        assert MSG in texts, row["id"]
    png = embed_text_png_bytes(cover, MSG, layout_id="r-bit0-msb")
    result = decode_bytes(png)
    assert result.found
    assert result.family == "lsb"
    assert result.text == MSG
    assert result.layout == "r-bit0-msb"


def test_png_text_container() -> None:
    cover = _cover(12)
    data = plant_png_text(cover, MSG, key="Comment")
    result = decode_bytes(data)
    assert result.found
    assert result.family == "container"
    assert result.text == MSG
    assert "tEXt" in (result.layout or "")


def test_jpeg_com_container() -> None:
    cover = _cover(13)
    data = plant_jpeg_com(cover, MSG)
    result = decode_bytes(data)
    assert result.found
    assert result.family == "container"
    assert result.text == MSG
    assert result.layout == "jpeg-COM"


def test_clean_cover_does_not_invent_text() -> None:
    cover = _cover(14, size=96)
    buf = BytesIO()
    Image.fromarray(cover).save(buf, format="PNG")
    result = decode_bytes(buf.getvalue())
    assert result.found is False
    assert result.family == "none"
    assert result.text is None


def test_jpeg_destroys_spatial_lsb() -> None:
    cover = _cover(15)
    marked = embed_text_array(cover, MSG, layout_id="r-bit0-msb")
    buf = BytesIO()
    Image.fromarray(marked).save(buf, format="JPEG", quality=90)
    result = decode_bytes(buf.getvalue())
    assert result.family != "lsb"
    assert result.found is False or result.family == "container"
    joined = " ".join(result.notes)
    assert "JPEG" in joined


def test_rgba_png_does_not_flatten_alpha() -> None:
    cover = _cover(16)
    marked = embed_text_array(cover, MSG, layout_id="r-bit0-msb")
    alpha = np.full(marked.shape[:2], 90, dtype=np.uint8)
    rgba = np.dstack([marked, alpha])
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    result = decode_bytes(buf.getvalue())
    assert result.found
    assert result.text == MSG
    assert result.family == "lsb"


def test_decode_json_schema_cli(tmp_path: Path) -> None:
    cover = _cover(17)
    src = tmp_path / "cover.png"
    out = tmp_path / "marked.png"
    Image.fromarray(cover).save(src)
    r0 = runner.invoke(app, ["embed-text", str(src), str(out), "--message", MSG])
    assert r0.exit_code == 0, r0.output
    r1 = runner.invoke(app, ["decode", str(out), "--json", "--no-report"])
    assert r1.exit_code == 0, r1.output
    assert '"found": true' in r1.stdout
    assert '"schema_version": 1' in r1.stdout
    assert MSG in r1.stdout
    got = decode_path(out)
    assert set(got.to_json()) >= {
        "schema_version",
        "found",
        "family",
        "text",
        "layout",
        "confidence",
        "notes",
        "candidates",
        "kind",
    }


def test_patch_lsb_found_without_coords() -> None:
    cover = _cover(21, size=192)
    marked = embed_text_patch(cover, MSG, x=90, y=40, w=70, h=42, layout_id="r-bit0-msb")
    buf = BytesIO()
    Image.fromarray(marked).save(buf, format="PNG")
    result = decode_bytes(buf.getvalue())
    assert result.found
    assert result.text == MSG
    assert result.hotspots, "blind tile hunt should flag the planted patch"
    xs = [h.x for h in result.hotspots]
    ys = [h.y for h in result.hotspots]
    assert any(abs(x - 90) <= 80 for x in xs)
    assert any(abs(y - 40) <= 80 for y in ys)


TOKEN = "INV_WM:LEFT_EYE:2026"


def test_token_scorer() -> None:
    assert looks_like_token(TOKEN)
    assert score_text(TOKEN, framed=True) >= 0.85
    assert score_text("H,S'H,s' iLQFiLAg", framed=True) == 0.0
    assert score_text("hello veil", framed=True) >= 0.85


def test_rgb_patch_token_without_coords() -> None:
    cover = _cover(22, size=192)
    marked = embed_text_patch(cover, TOKEN, x=11, y=13, w=70, h=42, layout_id="rgb-bit0-msb")
    buf = BytesIO()
    Image.fromarray(marked).save(buf, format="PNG")
    result = decode_bytes(buf.getvalue())
    assert result.found
    assert result.text == TOKEN


def test_wm_png_left_eye_token() -> None:
    photo = Path.home() / "Desktop" / "Wm .png"
    if not photo.is_file():
        return
    result = decode_path(photo)
    assert result.found
    assert result.text == TOKEN
    assert result.family == "lsb"


def test_decode_cli_writes_report_png(tmp_path: Path) -> None:
    cover = _cover(23, size=128)
    marked = embed_text_patch(cover, TOKEN, x=11, y=13, w=70, h=42, layout_id="rgb-bit0-msb")
    src = tmp_path / "marked.png"
    Image.fromarray(marked).save(src)
    dest = tmp_path / "out-report.png"
    r = runner.invoke(app, ["decode", str(src), "--out", str(dest)])
    assert r.exit_code == 0, r.output
    assert TOKEN in r.stdout
    assert dest.is_file()
    im = Image.open(dest)
    assert im.format == "PNG"
    assert im.size[0] >= 800
    assert im.size[1] >= 400
    r2 = runner.invoke(app, ["decode", str(src), "--no-report"])
    assert r2.exit_code == 0


def test_colleague_jpeg_identifies_eye_hotspot() -> None:
    photo = Path.home() / "Desktop" / "photo_2026-08-21_17-40-16.jpg"
    if not photo.is_file():
        return
    result = decode_path(photo)
    assert result.kind == "jpeg"
    assert result.hotspots, "should locate the left-eye LSB patch without coordinates"
    near = [
        h
        for h in result.hotspots
        if abs(h.x - 545) <= 96 and abs(h.y - 415) <= 96
    ]
    assert near, f"eye box missing from hotspots {[h.to_json() for h in result.hotspots]}"


def test_embed_text_path_png_text(tmp_path: Path) -> None:
    cover = _cover(18)
    src = tmp_path / "c.png"
    out = tmp_path / "t.png"
    Image.fromarray(cover).save(src)
    embed_text_path(src, out, MSG, family="png-text")
    result = decode_path(out)
    assert result.found
    assert result.family == "container"
    assert result.text == MSG
