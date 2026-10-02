"""Hunt HUD PNG. Same ink as decode dossier. ASCII only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from veilscan.decode.report import (
    CYAN,
    INK,
    INK2,
    MUTED,
    PAPER,
    PHOS,
    RED,
    _font,
    _mono,
    _wrap,
)
from veilscan.hunt.types import HuntResult


def default_hunt_dir(source: str | Path) -> Path:
    p = Path(source)
    raw = p.stem.strip() or "veilscan"
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw).strip("-_") or "veilscan"
    return p.with_name(f"{stem}-veilscan-hunt")


def _thumb(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = im.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return im.resize((nw, nh), Image.Resampling.LANCZOS)


def render_hunt_report(
    result: HuntResult,
    dest: Path,
    *,
    rgb: np.ndarray | None = None,
    bitplane_path: Path | None = None,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    photo = None
    if rgb is not None and getattr(rgb, "size", 0):
        arr = np.clip(rgb, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            photo = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            photo = Image.fromarray(arr[..., :3]).convert("RGB")
        photo = _thumb(photo, 720, 720)

    planes = None
    if bitplane_path is not None and Path(bitplane_path).is_file():
        try:
            planes = _thumb(Image.open(bitplane_path).convert("RGB"), 720, 280)
        except Exception:
            planes = None

    left_w = 0
    left_h = 0
    if photo is not None:
        left_w = max(left_w, photo.size[0])
        left_h += photo.size[1]
    if planes is not None:
        left_w = max(left_w, planes.size[0])
        left_h += planes.size[1] + (18 if photo is not None else 0)
    if left_w:
        left_w += 44
        left_h += 44

    dossier_w = 720
    margin = 22
    canvas_w = max(left_w + dossier_w + margin, dossier_w + margin * 2)
    canvas_h = max(left_h, 980, 120 + 28 * (8 + min(12, len(result.findings))))
    canvas = Image.new("RGB", (canvas_w, canvas_h), INK)
    grid = Image.new("RGB", (dossier_w, canvas_h), INK2)
    gd = ImageDraw.Draw(grid)
    for x in range(0, dossier_w, 28):
        gd.line((x, 0, x, canvas_h), fill=(16, 26, 40), width=1)
    for y in range(0, canvas_h, 28):
        gd.line((0, y, dossier_w, y), fill=(16, 26, 40), width=1)
    canvas.paste(grid, (canvas_w - dossier_w, 0))

    y_left = margin
    x_left = margin
    if photo is not None:
        canvas.paste(photo, (x_left, y_left))
        y_left += photo.size[1] + 16
    if planes is not None:
        canvas.paste(planes, (x_left, y_left))

    dd = ImageDraw.Draw(canvas)
    dx = canvas_w - dossier_w + 32
    dy = 32
    dw = dossier_w - 64
    f_disp = _font("display", 40)
    f_k = _font("sansb", 13)
    f_b = _font("satb", 20)
    f_body = _font("sans", 15)
    f_m = _mono(14)

    dd.text((dx, dy), "VEILSCAN", font=f_disp, fill=PAPER)
    dd.text((dx, dy + 46), "HUNT BRIEF", font=f_k, fill=CYAN)
    dy += 82

    status = result.status()
    badge = "FLAGS" if status == "flags" else ("HITS" if status == "findings" else "EMPTY")
    badge_col = PHOS if status == "flags" else (MUTED if status == "empty" else CYAN)
    tw = dd.textlength(badge, font=f_k) + 24
    dd.rounded_rectangle((dx, dy, dx + tw, dy + 26), radius=8, fill=badge_col)
    dd.text((dx + 10, dy + 5), badge, font=f_k, fill=INK)
    dd.text((dx + tw + 16, dy + 4), f"v{ _ver() }  {result.kind.upper()}", font=f_k, fill=MUTED)
    dy += 42

    dd.text((dx, dy), result.summary(), font=f_b, fill=PAPER)
    dy += 36
    meta = f"sha256 {result.sha256[:16]}...  {result.size} B  {result.elapsed_ms:.0f} ms"
    dd.text((dx, dy), meta, font=f_m, fill=MUTED)
    dy += 28

    dd.text((dx, dy), "FLAGS", font=f_k, fill=CYAN)
    dy += 22
    flag_box_h = 28 + 22 * max(1, min(6, len(result.flags) or 1))
    dd.rounded_rectangle((dx, dy, dx + dw, dy + flag_box_h), radius=8, outline=CYAN, width=1)
    fy = dy + 10
    if result.flags:
        for fl in result.flags[:6]:
            for line in _wrap(fl, f_m, dw - 24, dd)[:2]:
                dd.text((dx + 12, fy), line, font=f_m, fill=PHOS)
                fy += 20
    else:
        dd.text((dx + 12, fy), "(none)", font=f_m, fill=RED)
    dy += flag_box_h + 18

    dd.text((dx, dy), "FINDINGS", font=f_k, fill=CYAN)
    dy += 22
    rows = result.findings[:12]
    if not rows:
        dd.text((dx, dy), "No extract hits.", font=f_body, fill=MUTED)
        dy += 24
    for f in rows:
        mark = "FLAG" if f.flag_hit else "hit"
        col = PHOS if f.flag_hit else MUTED
        head = f"[{mark}] {f.family}/{f.method}"
        dd.text((dx, dy), head[:72], font=f_m, fill=col)
        dy += 18
        preview = (f.text or f.evidence or "").replace("\n", " ")
        if preview:
            dd.text((dx + 8, dy), preview[:78], font=f_body, fill=PAPER)
            dy += 20
        if dy > canvas_h - 80:
            break

    arts = [a for a in result.artifacts if a not in {"findings.json", "report.txt", "report.png"}]
    if arts:
        dy += 8
        dd.text((dx, dy), "ARTIFACTS  " + ", ".join(arts[:8]), font=f_m, fill=MUTED)

    dd.text((dx, canvas_h - 36), "read the veil, never a lift", font=_mono(13), fill=MUTED)
    canvas.save(dest, "PNG", optimize=True)
    return dest


def _ver() -> str:
    try:
        from veilscan import __version__

        return __version__
    except Exception:
        return "2.6.0"
