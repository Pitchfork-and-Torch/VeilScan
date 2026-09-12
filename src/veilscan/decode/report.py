"""Default decode artifact: HUD overlay plus executive dossier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from veilscan.decode.hotspots import Hotspot, cluster_hotspots
from veilscan.decode.score import looks_like_token

INK = (8, 12, 18)
INK2 = (12, 18, 28)
PAPER = (232, 236, 242)
PHOS = (206, 255, 74)
COBALT = (28, 92, 255)
CYAN = (90, 230, 255)
MUTED = (140, 168, 190)
GOLD = (255, 176, 48)
RED = (255, 72, 96)

PALETTE = [
    (255, 72, 148),
    (255, 176, 48),
    (48, 214, 224),
    (80, 220, 110),
    (188, 92, 255),
    (255, 128, 64),
    (255, 86, 186),
    (56, 168, 236),
    (255, 220, 64),
    (64, 255, 196),
    (255, 96, 96),
    (160, 200, 255),
]

KIT = Path.home() / "design-assets" / "fontshare"


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = {
        "display": KIT / "clash-display" / "otf" / "ClashDisplay-Bold.otf",
        "display2": KIT / "clash-display" / "otf" / "ClashDisplay-Semibold.otf",
        "sans": KIT / "general-sans" / "ttf" / "GeneralSans-Medium.ttf",
        "sansb": KIT / "general-sans" / "ttf" / "GeneralSans-Bold.ttf",
        "sat": KIT / "satoshi" / "otf" / "Satoshi-Medium.otf",
        "satb": KIT / "satoshi" / "otf" / "Satoshi-Bold.otf",
    }
    path = names.get(kind)
    if path and path.is_file():
        return ImageFont.truetype(str(path), size)
    for fallback in ("segoeui.ttf", "arial.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(fallback, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _mono(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("cascadiamono.ttf", "consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return _font("sans", size)


def _wrap(text: str, font, max_w: int, draw: ImageDraw.ImageDraw) -> list[str]:
    if not text:
        return [""]
    words = text.split(" ")
    if len(words) == 1 and draw.textlength(text, font=font) > max_w:
        lines: list[str] = []
        buf = ""
        for ch in text:
            trial = buf + ch
            if buf and draw.textlength(trial, font=font) > max_w:
                lines.append(buf)
                buf = ch
            else:
                buf = trial
        if buf:
            lines.append(buf)
        return lines
    lines = []
    cur = ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if cur and draw.textlength(trial, font=font) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _explain(text: str | None, family: str, layout: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if not text:
        rows.append(("verdict", "No keyless plaintext in this file."))
        rows.append(("limit", "Encrypted stego, vendor IDs, and neural marks stay silent."))
        return rows
    if looks_like_token(text):
        parts = text.split(":")
        labels = ["tag", "region", "stamp", "field 4", "field 5"]
        rows.append(("kind", "Structured token watermark"))
        for i, part in enumerate(parts):
            lab = labels[i] if i < len(labels) else f"field {i + 1}"
            rows.append((lab, part))
    else:
        rows.append(("kind", "Keyless plaintext"))
        rows.append(("body", text))
    rows.append(("carrier", f"{family} / {layout}"))
    return rows


def render_decode_report(
    rgb: np.ndarray,
    result,
    source: str | Path,
    out: str | Path,
    scan=None,
) -> Path:
    """Write a two-pane forensic PNG. Left: HUD photo. Right: executive brief."""
    src = Path(source)
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)

    photo = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).convert("RGB")
    pw, ph = photo.size
    max_h, max_w = 1400, 1120
    scale = min(max_h / ph, max_w / pw, 1.0)
    nw, nh = max(1, int(pw * scale)), max(1, int(ph * scale))
    photo = photo.resize((nw, nh), Image.Resampling.LANCZOS)
    sx, sy = nw / pw, nh / ph

    dossier_w = 680
    margin = 22
    canvas_w = nw + dossier_w + margin * 3
    canvas_h = max(nh + margin * 2, 1180)
    canvas = Image.new("RGB", (canvas_w, canvas_h), INK)
    # faint dossier grid
    grid = Image.new("RGB", (dossier_w, canvas_h), INK2)
    gd = ImageDraw.Draw(grid)
    for x in range(0, dossier_w, 28):
        gd.line((x, 0, x, canvas_h), fill=(16, 26, 40), width=1)
    for y in range(0, canvas_h, 28):
        gd.line((0, y, dossier_w, y), fill=(16, 26, 40), width=1)
    canvas.paste(grid, (nw + margin * 2, 0))
    canvas.paste(photo, (margin, margin))

    overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    ox, oy = margin, margin

    spots = list(result.hotspots[:12])
    groups = cluster_hotspots(spots)
    f_hud = _mono(15)
    f_tiny = _mono(13)
    f_badge = _font("sansb", 13)

    # cluster hulls first
    for g in groups:
        if len(g) < 2:
            continue
        xs = [int(h.x * sx) + ox for h in g]
        ys = [int(h.y * sy) + oy for h in g]
        x1 = [int((h.x + h.w) * sx) + ox for h in g]
        y1 = [int((h.y + h.h) * sy) + oy for h in g]
        pad = 10
        box = (min(xs) - pad, min(ys) - pad, max(x1) + pad, max(y1) + pad)
        d.rounded_rectangle(box, radius=8, outline=GOLD + (200,), width=2)
        ids = " / ".join(f"#{spots.index(h) + 1}" for h in g if h in spots)
        d.rectangle((box[0], box[1] - 22, box[0] + 8 + 7 * (10 + len(ids)), box[1]), fill=(40, 24, 8, 210))
        d.text((box[0] + 6, box[1] - 20), f"CLUSTER {ids}", font=f_tiny, fill=GOLD)

    clustered: set[int] = set()
    for g in groups:
        if len(g) >= 2:
            for h in g:
                if h in spots:
                    clustered.add(id(h))

    for i, h in enumerate(spots):
        color = PALETTE[i % len(PALETTE)]
        x0 = int(h.x * sx) + ox
        y0 = int(h.y * sy) + oy
        x1 = int((h.x + h.w) * sx) + ox
        y1 = int((h.y + h.h) * sy) + oy
        d.rounded_rectangle((x0, y0, x1, y1), radius=3, outline=color + (230,), width=2)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        d.line((cx - 7, cy, cx + 7, cy), fill=color + (240,), width=2)
        d.line((cx, cy - 7, cx, cy + 7), fill=color + (240,), width=2)
        bw, bh = 18, 16
        d.rounded_rectangle((x0 - 1, y0 - bh - 2, x0 + bw, y0 - 1), radius=3, fill=color + (240,))
        d.text((x0 + 4, y0 - bh), str(i + 1), font=f_badge, fill=(8, 8, 12, 255))
        if id(h) in clustered and i > 2:
            continue
        label = f"#{i + 1}  ({h.x},{h.y})  {h.w}x{h.h}  corr={h.corr:.3f}"
        lx = x1 + 8
        ly = y0
        tw = d.textlength(label, font=f_tiny)
        if lx + tw + 12 > ox + nw:
            lx = max(ox + 8, x0 - int(tw) - 16)
        d.rounded_rectangle((lx - 4, ly - 2, lx + tw + 8, ly + 16), radius=3, fill=(6, 10, 16, 200))
        d.text((lx, ly), label, font=f_tiny, fill=color + (255,))

    # payload pin
    payload_box = None
    if result.found and result.candidates:
        bb = result.candidates[0].bbox
        if bb and bb[2] > 1:
            payload_box = bb
    if payload_box:
        x, y, w, h = payload_box
        if h <= 12:
            w = max(w, 70)
            h = 42
        x0 = int(x * sx) + ox
        y0 = int(y * sy) + oy
        x1 = int((x + w) * sx) + ox
        y1 = int((y + h) * sy) + oy
        d.rounded_rectangle((x0 - 4, y0 - 4, x1 + 4, y1 + 4), radius=5, outline=PHOS + (255,), width=3)
        d.rounded_rectangle((x0, y0, x1, y1), radius=3, outline=PHOS + (120,), width=1)
        tag_w = 124
        tag_y = max(oy + 4, y0 - 22)
        d.rounded_rectangle((x0, tag_y, x0 + tag_w, tag_y + 18), radius=3, fill=(20, 32, 8, 235))
        d.text((x0 + 8, tag_y + 2), "HIDDEN TEXT", font=f_tiny, fill=PHOS)

    # compact HUD on the photo (the format the operator liked)
    hud_x, hud_y = ox + 14, oy + 14
    hud_w = min(420, nw - 28)
    lines = [
        "VEILSCAN   LSB HOTSPOTS",
        f"source: {src.name}   {pw}x{ph}   overlay of scan receipt",
    ]
    if scan is not None:
        present = "YES" if getattr(scan, "present", False) else "NO"
        hint = getattr(scan, "family_hint", "none")
        sc = float(getattr(scan, "score", 0.0) or 0.0)
        lines.append(f"scan: present={present}  hint={hint}  score={sc:.3f}")
    if result.found:
        lines.append(
            f"FOUND  family={result.family}  layout={result.layout}  conf={result.confidence:.3f}"
        )
        lines.append(f"decode: {result.text}")
    else:
        lines.append("NO KEYLESS PLAINTEXT")
    lines.append("note: blind 64x64 tile hunt, no prior box")
    for i, h in enumerate(spots[:8]):
        lines.append(f"#{i + 1}  x={h.x}  y={h.y}  {h.w}x{h.h}  corr={h.corr:.3f}")
    line_h = 16
    hud_h = 16 + line_h * len(lines)
    d.rounded_rectangle(
        (hud_x, hud_y, hud_x + hud_w, hud_y + hud_h),
        radius=8,
        fill=(6, 16, 28, 210),
        outline=CYAN + (180,),
        width=1,
    )
    yy = hud_y + 8
    d.text((hud_x + 12, yy), lines[0], font=_font("sansb", 14), fill=CYAN + (255,))
    yy += line_h
    for line in lines[1:]:
        col = PHOS + (255,) if line.startswith("FOUND") or line.startswith("decode:") else MUTED + (255,)
        if line.startswith("NO "):
            col = RED + (255,)
        if line.startswith("#"):
            idx = int(line[1:].split()[0]) - 1
            col = PALETTE[idx % len(PALETTE)] + (255,)
        d.text((hud_x + 12, yy), line, font=f_hud, fill=col)
        yy += line_h

    # footer on photo
    foot = f"VeilScan blind LSB tile hunt  |  {src.name}  |  64 px tiles"
    d.rectangle((ox, oy + nh - 26, ox + nw, oy + nh), fill=(6, 10, 16, 200))
    d.text((ox + 12, oy + nh - 20), foot, font=f_tiny, fill=CYAN + (220,))

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    dd = ImageDraw.Draw(canvas)

    # ----- dossier -----
    dx = nw + margin * 2 + 28
    dy = 36
    dw = dossier_w - 56
    f_disp = _font("display", 42)
    f_k = _font("sansb", 13)
    f_b = _font("satb", 22)
    f_body = _font("sans", 16)
    f_m = _mono(16)
    f_m2 = _mono(20)

    dd.text((dx, dy), "VEILSCAN", font=f_disp, fill=PAPER)
    dd.text((dx, dy + 48), "DECODE BRIEF", font=f_k, fill=CYAN)
    dy += 84
    dd.line((dx, dy, dx + dw, dy), fill=(40, 60, 90), width=1)
    dy += 18

    found = bool(result.found)
    pill = "FOUND" if found else "NO TEXT"
    pill_col = PHOS if found else RED
    pw_ = 18 + int(dd.textlength(pill, font=f_k))
    dd.rounded_rectangle((dx, dy, dx + pw_, dy + 26), radius=13, fill=pill_col)
    dd.text((dx + 9, dy + 6), pill, font=f_k, fill=INK)
    dd.text((dx + pw_ + 12, dy + 6), f"v{ _version() }   {result.kind.upper()}", font=f_tiny, fill=MUTED)
    dy += 42

    dd.text((dx, dy), "HIDDEN TEXT", font=f_k, fill=GOLD)
    dy += 22
    vault_h = 132
    dd.rounded_rectangle((dx, dy, dx + dw, dy + vault_h), radius=10, fill=(6, 10, 14), outline=PHOS if found else (60, 70, 80))
    body = result.text if found else "No keyless plaintext recovered."
    wrap = _wrap(body, f_m2, dw - 28, dd)
    ty = dy + 16
    for line in wrap[:5]:
        dd.text((dx + 14, ty), line, font=f_m2, fill=PHOS if found else MUTED)
        ty += 24
    dy += vault_h + 22

    dd.text((dx, dy), "WHAT THIS MEANS", font=f_k, fill=GOLD)
    dy += 20
    for lab, val in _explain(result.text if found else None, result.family, result.layout or "-"):
        dd.text((dx, dy), lab.upper(), font=f_tiny, fill=MUTED)
        vv = _wrap(val, f_body, dw - 100, dd)
        dd.text((dx + 92, dy), vv[0], font=f_body, fill=PAPER)
        dy += 22
        for extra in vv[1:]:
            dd.text((dx + 92, dy), extra, font=f_body, fill=PAPER)
            dy += 20
    dy += 10
    dd.line((dx, dy, dx + dw, dy), fill=(40, 60, 90), width=1)
    dy += 16

    dd.text((dx, dy), "METHOD", font=f_k, fill=GOLD)
    dy += 20
    method_bits = [
        f"family   {result.family}",
        f"layout   {result.layout or '-'}",
        f"conf     {result.confidence:.3f}",
        f"pixels   {pw} x {ph}",
    ]
    for line in method_bits:
        dd.text((dx, dy), line, font=f_m, fill=CYAN)
        dy += 20
    dy += 8

    dd.text((dx, dy), "HOTSPOTS", font=f_k, fill=GOLD)
    dy += 18
    if not spots:
        dd.text((dx, dy), "none", font=f_m, fill=MUTED)
        dy += 20
    for i, h in enumerate(spots[:8]):
        col = PALETTE[i % len(PALETTE)]
        dd.rectangle((dx, dy + 4, dx + 10, dy + 14), fill=col)
        dd.text(
            (dx + 16, dy),
            f"#{i + 1}  ({h.x},{h.y})  {h.w}x{h.h}  corr={h.corr:.3f}",
            font=f_m,
            fill=col,
        )
        dy += 18
        if dy > canvas_h - 90:
            break

    dy += 8
    if dy < canvas_h - 70:
        dd.text((dx, dy), "LIMITS", font=f_k, fill=GOLD)
        dy += 18
        limit = (
            "Keyless plaintext only. JPEG often kills spatial LSB. "
            "A hotspot is not proof of a message. A message is not a copyright ruling."
        )
        for line in _wrap(limit, f_body, dw, dd)[:4]:
            dd.text((dx, dy), line, font=f_body, fill=MUTED)
            dy += 20

    dd.text((dx, canvas_h - 36), "read the veil, never a lift", font=f_tiny, fill=MUTED)
    canvas.save(dest, "PNG", optimize=True)
    return dest


def _version() -> str:
    try:
        from veilscan import __version__

        return __version__
    except Exception:
        return "2.0.0"


def default_report_path(source: str | Path) -> Path:
    p = Path(source)
    raw = p.stem.strip() or "veilscan"
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw).strip("-_") or "veilscan"
    return p.with_name(f"{stem}-veilscan-report.png")
