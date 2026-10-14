"""WAV PCM LSB and spectrogram stills.

Eval plants live in gym. This module extracts. It does not hide for operators.
Letter OCR runs only when `tesseract` is on PATH.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from veilscan.decode.lsb import message_to_bits, pack_bits
from veilscan.hunt.flags import find_flags
from veilscan.hunt.payloads import inspect_payload
from veilscan.hunt.types import HuntFinding

TONE_MAGIC = b"VSPEC1"
TONE_BIN0 = 16
TONE_NPERSEG = 256
QR_NPERSEG = 256
_MAX_SAMPLES = 2_000_000


@dataclass
class WavPcm:
    samples: np.ndarray  # (n, channels) int16 or uint8
    sample_rate: int
    bits: int
    channels: int


@dataclass
class AudioPass:
    findings: list[HuntFinding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


def parse_wav(data: bytes) -> WavPcm | None:
    """PCM 8/16-bit, 1 or 2 channels. None when the RIFF is not that."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos = 12
    n = len(data)
    fmt: tuple[int, int, int, int, int] | None = None
    pcm: bytes | None = None
    while pos + 8 <= n:
        cid = data[pos : pos + 4]
        ln = struct.unpack_from("<I", data, pos + 4)[0]
        start = pos + 8
        end = start + ln
        if end > n:
            break
        chunk = data[start:end]
        if cid == b"fmt " and len(chunk) >= 16:
            audio_fmt, ch, sr, _rate, block, bits = struct.unpack_from("<HHIIHH", chunk)
            fmt = (audio_fmt, ch, sr, block, bits)
        elif cid == b"data" and pcm is None:
            pcm = chunk
        pos = end + (ln & 1)
    if fmt is None or pcm is None:
        return None
    audio_fmt, ch, sr, _block, bits = fmt
    if audio_fmt != 1 or ch not in (1, 2) or bits not in (8, 16) or sr <= 0:
        return None
    if bits == 16:
        usable = (len(pcm) // (2 * ch)) * (2 * ch)
        samples = np.frombuffer(pcm[:usable], dtype="<i2").reshape(-1, ch).copy()
    else:
        usable = (len(pcm) // ch) * ch
        samples = np.frombuffer(pcm[:usable], dtype=np.uint8).reshape(-1, ch).copy()
    if samples.shape[0] > _MAX_SAMPLES:
        samples = samples[:_MAX_SAMPLES].copy()
    return WavPcm(samples=samples, sample_rate=int(sr), bits=int(bits), channels=int(ch))


def encode_wav(samples: np.ndarray, sample_rate: int, bits: int) -> bytes:
    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    arr = np.ascontiguousarray(arr)
    channels = int(arr.shape[1])
    if bits == 16:
        raw = np.ascontiguousarray(arr.astype("<i2")).tobytes()
        block = channels * 2
        bps = 16
    elif bits == 8:
        raw = np.ascontiguousarray(arr.astype(np.uint8)).tobytes()
        block = channels
        bps = 8
    else:
        raise ValueError("bits must be 8 or 16")
    fmt = struct.pack("<HHIIHH", 1, channels, int(sample_rate), int(sample_rate) * block, block, bps)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(raw)) + raw
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _write_low_bits(seq: np.ndarray, bits: np.ndarray, bit: int) -> None:
    n = int(min(seq.size, bits.size))
    if n <= 0:
        return
    piece = seq[:n]
    vals = piece.astype(np.uint16, copy=True)
    mask = np.uint16(0xFFFF ^ (1 << bit))
    vals = (vals & mask) | (bits[:n].astype(np.uint16) << np.uint16(bit))
    if piece.dtype == np.uint8:
        piece[:] = vals.astype(np.uint8)
    else:
        piece[:] = vals.astype(np.int16)


def _walk(samples: np.ndarray, channel: int | None, reverse: bool) -> np.ndarray:
    seq = samples.reshape(-1) if channel is None else samples[:, channel]
    if reverse:
        seq = seq[::-1]
    return seq


def plant_wav_lsb(
    flag: str,
    *,
    bits: int = 16,
    channels: int = 1,
    channel: int | None = 0,
    reverse: bool = False,
    sample_rate: int = 8000,
    bit: int = 0,
) -> bytes:
    """Eval-only. Plants framed LSB on silence so unused samples stay quiet."""
    payload = message_to_bits(flag, msb_first=True)
    n = int(payload.size + 64)
    if bits == 8:
        samples = np.full((n, channels), 128, dtype=np.uint8)
    else:
        samples = np.zeros((n, channels), dtype=np.int16)
    _write_low_bits(_walk(samples, channel, reverse), payload, bit)
    return encode_wav(samples, sample_rate, bits)


def _frame_mag(samples: np.ndarray, nperseg: int) -> np.ndarray | None:
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0:
        return None
    if x.size < nperseg:
        x = np.pad(x, (0, int(nperseg - x.size)))
    n = (x.size // nperseg) * nperseg
    frames = x[:n].reshape(-1, nperseg)
    # Rectangular window on purpose: plant tones sit on exact bins.
    spec = np.fft.rfft(frames, axis=1)
    return np.abs(spec).T


def _tone_frame(byte: int) -> np.ndarray:
    t = np.arange(TONE_NPERSEG, dtype=np.float64)
    sig = np.zeros(TONE_NPERSEG, dtype=np.float64)
    for i in range(8):
        if (byte >> (7 - i)) & 1:
            k = TONE_BIN0 + i
            sig += np.sin(2.0 * np.pi * k * t / TONE_NPERSEG)
    peak = float(np.max(np.abs(sig))) or 1.0
    return sig / peak


def plant_wav_tone(flag: str, *, sample_rate: int = 8000) -> bytes:
    """Eval-only. Each byte is 8 exact-bin tones, headed by VSPEC1."""
    body = TONE_MAGIC + flag.encode("utf-8") + b"\x00"
    frames = [_tone_frame(b) for b in body]
    audio = np.concatenate(frames)
    samples = np.rint(audio * 14000.0).astype(np.int16).reshape(-1, 1)
    return encode_wav(samples, sample_rate, 16)


def _extract_tone(samples: np.ndarray) -> bytes | None:
    mag = _frame_mag(samples, TONE_NPERSEG)
    if mag is None or mag.shape[0] <= TONE_BIN0 + 8:
        return None
    band = mag[TONE_BIN0 : TONE_BIN0 + 8, :]
    peak = float(band.max()) if band.size else 0.0
    if peak <= 0:
        return None
    out = bytearray()
    for col in range(band.shape[1]):
        colv = band[:, col]
        if float(colv.max()) < 0.25 * peak:
            out.append(0)
            continue
        thr = 0.45 * float(colv.max())
        byte = 0
        for i in range(8):
            if float(colv[i]) >= thr:
                byte |= 1 << (7 - i)
        out.append(byte)
    raw = bytes(out)
    idx = raw.find(TONE_MAGIC)
    if idx < 0:
        return None
    rest = raw[idx + len(TONE_MAGIC) :]
    end = rest.find(b"\x00")
    payload = rest if end < 0 else rest[:end]
    return payload[:512]


def _qr_black(flag: str) -> np.ndarray | None:
    try:
        import cv2
    except Exception:
        return None
    enc = cv2.QRCodeEncoder.create()
    qr = np.asarray(enc.encode(flag))
    if qr.ndim != 2 or qr.size == 0:
        return None
    black = qr == 0
    # A real code is not almost-all-black and not almost-all-white.
    if black.mean() > 0.85 or black.mean() < 0.05:
        black = qr > 0
    q = 4
    h, w = black.shape
    canvas = np.zeros((h + 2 * q, w + 2 * q), dtype=bool)
    canvas[q : q + h, q : q + w] = black
    return canvas


def plant_wav_qr(flag: str, *, sample_rate: int = 8000) -> bytes | None:
    """Eval-only. Black QR modules become exact-bin tones. None if OpenCV cannot encode."""
    canvas = _qr_black(flag)
    if canvas is None:
        return None
    h, w = canvas.shape
    if 8 + h >= QR_NPERSEG // 2:
        return None
    t = np.arange(QR_NPERSEG, dtype=np.float64)
    frames: list[np.ndarray] = []
    for col in range(w):
        sig = np.zeros(QR_NPERSEG, dtype=np.float64)
        for row in range(h):
            if not canvas[row, col]:
                continue
            # Fixed amplitude per module so a dense column is not quieter per bin.
            k = 8 + row
            sig += np.sin(2.0 * np.pi * k * t / QR_NPERSEG)
        frames.append(sig)
    audio = np.concatenate(frames)
    peak = float(np.max(np.abs(audio))) or 1.0
    samples = np.rint(audio / peak * 12000.0).astype(np.int16).reshape(-1, 1)
    return encode_wav(samples, sample_rate, 16)


def _decode_qr_texts(image_u8: np.ndarray) -> list[str]:
    try:
        import cv2
    except Exception:
        return []
    if image_u8.ndim != 2 or min(image_u8.shape) < 16:
        return []
    det = cv2.QRCodeDetector()
    found: list[str] = []
    seen: set[str] = set()
    h, w = image_u8.shape
    for inv in (False, True):
        base = 255 - image_u8 if inv else image_u8
        for sc in (4, 8, 2):
            im = cv2.resize(base, (w * sc, h * sc), interpolation=cv2.INTER_NEAREST)
            try:
                val, _pts, _ = det.detectAndDecode(im)
            except Exception:
                continue
            if val and val not in seen:
                seen.add(val)
                found.append(val)
    return found


def _extract_qr(samples: np.ndarray) -> list[str]:
    mag = _frame_mag(samples, QR_NPERSEG)
    if mag is None:
        return []
    band = mag[1:-1, :]
    if band.size == 0 or float(band.max()) <= 0:
        return []
    ref = float(np.percentile(band, 99))
    if ref <= 0:
        return []
    ink = np.zeros(band.shape, dtype=bool)
    for col in range(band.shape[1]):
        colv = band[:, col]
        if float(colv.max()) < 0.2 * ref:
            continue
        thr = 0.45 * float(colv.max())
        ink[:, col] = colv >= thr
    if not np.any(ink):
        return []
    ys, xs = np.where(ink)
    y0 = max(int(ys.min()) - 4, 0)
    y1 = min(int(ys.max()) + 5, ink.shape[0])
    x0 = max(int(xs.min()) - 4, 0)
    x1 = min(int(xs.max()) + 5, ink.shape[1])
    crop = ink[y0:y1, x0:x1]
    if min(crop.shape) < 17:
        return []
    dark_on_white = np.where(crop, 0, 255).astype(np.uint8)
    return _decode_qr_texts(dark_on_white)


def spectrogram_u8(samples: np.ndarray, nperseg: int = 256) -> np.ndarray | None:
    """Log magnitude, DC dropped, low frequency at the bottom."""
    mag = _frame_mag(samples, nperseg)
    if mag is None:
        return None
    view = np.log1p(mag[1:, :])
    view = view - float(view.min())
    peak = float(view.max()) or 1.0
    u8 = np.clip(view / peak * 255.0, 0, 255).astype(np.uint8)
    return u8[::-1]


def render_spectrogram_png(samples: np.ndarray) -> bytes | None:
    u8 = spectrogram_u8(samples, 256)
    if u8 is None:
        return None
    im = Image.fromarray(u8, mode="L")
    im = im.resize((max(320, im.width * 3), max(180, im.height * 3)), Image.Resampling.NEAREST)
    import io

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _lsb_findings(pcm: WavPcm, cre, *, deep: bool, stop_on_flag: bool) -> list[HuntFinding]:
    samples = pcm.samples
    bits = range(3) if deep else range(1)
    walks: list[tuple[str, int | None]] = [(f"ch{i}", i) for i in range(pcm.channels)]
    if pcm.channels > 1:
        walks.append(("interleaved", None))
    seen: set[str] = set()
    out: list[HuntFinding] = []
    width = "pcm8" if pcm.bits == 8 else "pcm16"
    for walk_name, channel in walks:
        for bit in bits:
            for reverse in (False, True):
                seq = _walk(samples, channel, reverse)
                packed_bits = ((seq.astype(np.uint16) >> bit) & 1).astype(np.uint8)
                if packed_bits.size < 16:
                    continue
                for msb_first in (True, False):
                    raw = pack_bits(packed_bits, msb_first=msb_first)
                    if not raw:
                        continue
                    direction = "rev" if reverse else "fwd"
                    order = "msb" if msb_first else "lsb"
                    method = f"{width}-{walk_name}-b{bit}-{direction}-{order}"
                    for hit in inspect_payload(raw, cre):
                        flags = list(hit.get("flags") or [])
                        if not flags:
                            continue
                        text = str(hit.get("text") or "")
                        key = text[:180] + method
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(
                            HuntFinding(
                                family="audio",
                                method=method,
                                confidence=float(hit.get("confidence") or 0.9),
                                evidence="pcm sample LSB",
                                text=text or None,
                                flag_hit=True,
                                extra={"flags": flags, "kind": hit.get("kind")},
                            )
                        )
                        if stop_on_flag or len(out) >= 8:
                            return out
    return out


def _glyph_bitmap(text: str) -> np.ndarray:
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default(size=42)
    probe = Image.new("L", (8, 8))
    draw = ImageDraw.Draw(probe)
    box = draw.textbbox((0, 0), text, font=font)
    width = max(8, int(box[2] - box[0]) + 16)
    height = max(8, int(box[3] - box[1]) + 12)
    im = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(im)
    draw.text((8 - box[0], 6 - box[1]), text, fill=255, font=font)
    arr = np.asarray(im, dtype=np.uint8)
    if arr.shape[0] > 96:
        im = im.resize((max(8, int(arr.shape[1] * 96 / arr.shape[0])), 96), Image.Resampling.NEAREST)
        arr = np.asarray(im, dtype=np.uint8)
    return np.where(arr > 48, np.uint8(255), np.uint8(0))


def plant_wav_glyphs(flag: str, *, sample_rate: int = 8000) -> bytes:
    """Eval-only. Letters exist as exact-bin ink. The file bytes do not hold the text."""
    bitmap = _glyph_bitmap(flag)
    height, width = bitmap.shape
    nperseg = 256
    t = np.arange(nperseg, dtype=np.float64)
    frames: list[np.ndarray] = []
    for col in range(-2, width + 2):
        sig = np.zeros(nperseg, dtype=np.float64)
        if 0 <= col < width:
            rows = np.flatnonzero(bitmap[:, col] > 0)
            for row in rows.tolist():
                k = 110 - int(row)
                if 8 <= k < nperseg // 2:
                    sig += np.sin(2.0 * np.pi * k * t / nperseg)
            peak = float(np.max(np.abs(sig))) or 1.0
            sig = sig / peak
        frames.append(sig)
    audio = np.concatenate(frames)
    samples = np.rint(audio * 16000.0).astype(np.int16).reshape(-1, 1)
    return encode_wav(samples, sample_rate, 16)


def _ocr_line(samples: np.ndarray) -> str | None:
    import shutil
    import subprocess

    exe = shutil.which("tesseract")
    if not exe:
        return None
    u8 = spectrogram_u8(samples, 256)
    if u8 is None or u8.size == 0 or int(u8.max()) == 0:
        return None
    ink = u8 >= 64
    if not np.any(ink):
        return None
    ys, xs = np.where(ink)
    y0 = max(int(ys.min()) - 2, 0)
    y1 = min(int(ys.max()) + 3, ink.shape[0])
    x0 = max(int(xs.min()) - 2, 0)
    x1 = min(int(xs.max()) + 3, ink.shape[1])
    # Hard threshold. Weak spectral leakage is not letter ink.
    dark = np.where(ink[y0:y1, x0:x1], np.uint8(0), np.uint8(255))
    im = Image.fromarray(dark, mode="L")
    im = im.resize((max(1, im.width * 4), max(1, im.height * 4)), Image.Resampling.NEAREST)
    padded = Image.new("L", (im.width + 40, im.height + 40), 255)
    padded.paste(im, (20, 20))
    import io

    buf = io.BytesIO()
    padded.save(buf, format="PNG")
    try:
        proc = subprocess.run(
            [
                exe,
                "stdin",
                "stdout",
                "--psm",
                "7",
                "-c",
                "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}-",
            ],
            input=buf.getvalue(),
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace")


def _ocr_findings(samples: np.ndarray, cre) -> list[HuntFinding]:
    raw = _ocr_line(samples)
    if not raw:
        return []
    text = "".join(ch for ch in raw if ch not in " \t\r\n")
    flags = find_flags(text, cre)
    if not flags:
        return []
    return [
        HuntFinding(
            family="audio",
            method="spectrogram-ocr",
            confidence=0.9,
            evidence="tesseract on the magnitude still",
            text=flags[0][:4096],
            flag_hit=True,
            extra={"flags": flags},
        )
    ]


def fill_pcm_findings(
    passed: AudioPass,
    pcm: WavPcm,
    cre,
    *,
    deep: bool = False,
    stop_on_flag: bool = False,
    dest: Path | None = None,
) -> AudioPass:
    """LSB, tone bytes, exact-bin QR, and PATH OCR for one PCM buffer."""
    if not any(note.startswith("Spectrogram still") for note in passed.notes):
        passed.notes.append(
            "Spectrogram still is magnitude. Letter OCR runs when tesseract is on PATH. "
            "Automated audio hits are PCM LSB, VSPEC1 tone bytes, a QR in frequency bins, and that OCR line."
        )
    png = render_spectrogram_png(pcm.samples[:, 0])
    if png and dest is not None and "spectrogram.png" not in passed.artifacts:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "spectrogram.png").write_bytes(png)
        passed.artifacts.append("spectrogram.png")

    passed.findings.extend(_lsb_findings(pcm, cre, deep=deep, stop_on_flag=stop_on_flag))
    if stop_on_flag and any(f.flag_hit for f in passed.findings):
        return passed

    tone = _extract_tone(pcm.samples[:, 0])
    if tone:
        try:
            text = tone.decode("utf-8")
        except UnicodeDecodeError:
            text = tone.decode("latin-1", "replace")
        flags = find_flags(text, cre)
        if flags or (text and text.isprintable() and len(text) >= 4):
            passed.findings.append(
                HuntFinding(
                    family="audio",
                    method="spectrogram-tone",
                    confidence=0.97 if flags else 0.8,
                    evidence="VSPEC1 exact-bin bytes",
                    text=text[:4096],
                    flag_hit=bool(flags),
                    extra={"flags": flags},
                )
            )
            if stop_on_flag and flags:
                return passed

    for text in _extract_qr(pcm.samples[:, 0]):
        flags = find_flags(text, cre)
        if not flags:
            continue
        passed.findings.append(
            HuntFinding(
                family="audio",
                method="spectrogram-qr",
                confidence=0.95,
                evidence="QR on exact-bin spectrogram",
                text=text[:4096],
                flag_hit=True,
                extra={"flags": flags},
            )
        )
        if stop_on_flag:
            break

    if not (stop_on_flag and any(f.flag_hit for f in passed.findings)):
        passed.findings.extend(_ocr_findings(pcm.samples[:, 0], cre))
    return passed


def audio_pass(
    data: bytes,
    cre,
    *,
    deep: bool = False,
    stop_on_flag: bool = False,
    dest: Path | None = None,
) -> AudioPass:
    passed = AudioPass()
    if not (data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WAVE"):
        return passed
    pcm = parse_wav(data)
    if pcm is None or pcm.samples.size == 0:
        passed.notes.append("RIFF WAVE is not PCM 8/16 mono or stereo. Sample LSB and spectrogram skipped.")
        return passed
    return fill_pcm_findings(passed, pcm, cre, deep=deep, stop_on_flag=stop_on_flag, dest=dest)
