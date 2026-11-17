"""MP3 and FLAC hunt.

Metadata is stdlib. PCM walks reuse the WAV sample path after an optional
miniaudio decode. MP3 is lossy, so sample LSB is not a gym claim. FLAC plants
use a verbatim subset so 16-bit samples round-trip. Eval plants only.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from veilscan.hunt.audio import WavPcm, fill_pcm_findings
from veilscan.hunt.flags import find_flags
from veilscan.hunt.types import HuntFinding

_BLOCK = 256


class _Bits:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.n = 0

    def write(self, value: int, nbits: int) -> None:
        if nbits <= 0:
            return
        value &= (1 << nbits) - 1
        for shift in range(nbits - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> shift) & 1)
            self.n += 1
            if self.n == 8:
                self.buf.append(self.acc)
                self.acc = 0
                self.n = 0

    def write_signed(self, value: int, nbits: int) -> None:
        if value < 0:
            value = (1 << nbits) + int(value)
        self.write(int(value), nbits)

    def bytes(self) -> bytes:
        if self.n:
            raise RuntimeError("bit writer is not byte-aligned")
        return bytes(self.buf)


def _crc8(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def _crc16(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b << 8
        for _ in range(8):
            if c & 0x8000:
                c = ((c << 1) ^ 0x8005) & 0xFFFF
            else:
                c = (c << 1) & 0xFFFF
    return c


def _utf8_int(n: int) -> bytes:
    if n < 0x80:
        return bytes((n,))
    if n < 0x800:
        return bytes((0xC0 | (n >> 6), 0x80 | (n & 0x3F)))
    if n < 0x10000:
        return bytes((0xE0 | (n >> 12), 0x80 | ((n >> 6) & 0x3F), 0x80 | (n & 0x3F)))
    return bytes(
        (
            0xF0 | (n >> 18),
            0x80 | ((n >> 12) & 0x3F),
            0x80 | ((n >> 6) & 0x3F),
            0x80 | (n & 0x3F),
        )
    )


def _streaminfo(sample_rate: int, total: int) -> bytes:
    bw = _Bits()
    bw.write(_BLOCK, 16)
    bw.write(_BLOCK, 16)
    bw.write(0, 24)
    bw.write(0, 24)
    bw.write(sample_rate, 20)
    bw.write(0, 3)  # mono
    bw.write(15, 5)  # 16-bit
    bw.write(total, 36)
    # 16 zero bytes: MD5 unknown. Decoders skip the check.
    return bw.bytes() + (b"\x00" * 16)


def _meta(block_type: int, body: bytes, last: bool) -> bytes:
    n = len(body)
    return bytes(((0x80 if last else 0) | (block_type & 0x7F), (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF)) + body


def _flac_frame(samples: np.ndarray, sample_number: int) -> bytes:
    header = bytes((0xFF, 0xF8, 0x80, 0x08)) + _utf8_int(sample_number)
    header += bytes((_crc8(header),))
    bw = _Bits()
    bw.write(0, 1)
    bw.write(1, 6)  # verbatim
    bw.write(0, 1)
    for sample in samples.astype(np.int16).tolist():
        bw.write_signed(int(sample), 16)
    body = header + bw.bytes()
    return body + struct.pack(">H", _crc16(body))


def _vorbis_comment(pairs: list[tuple[str, str]]) -> bytes:
    vendor = b"veilscan"
    parts = [struct.pack("<I", len(vendor)), vendor, struct.pack("<I", len(pairs))]
    for key, value in pairs:
        raw = f"{key}={value}".encode("utf-8")
        parts.append(struct.pack("<I", len(raw)))
        parts.append(raw)
    return b"".join(parts)


def encode_verbatim_flac(
    samples: np.ndarray,
    sample_rate: int = 8000,
    *,
    comments: list[tuple[str, str]] | None = None,
) -> bytes:
    """Eval-only subset FLAC. Mono 16-bit verbatim frames. Not a general encoder."""
    pcm = np.asarray(samples, dtype=np.int16).reshape(-1)
    if pcm.size == 0:
        pcm = np.zeros(1, dtype=np.int16)
    pad = (-int(pcm.size)) % _BLOCK
    if pad:
        pcm = np.pad(pcm, (0, pad))
    blocks = [_meta(0, _streaminfo(int(sample_rate), int(pcm.size)), last=not comments)]
    if comments:
        blocks.append(_meta(4, _vorbis_comment(comments), last=True))
    frames = []
    for i in range(0, int(pcm.size), _BLOCK):
        frames.append(_flac_frame(pcm[i : i + _BLOCK], i))
    return b"fLaC" + b"".join(blocks) + b"".join(frames)


def _syncsafe(raw: bytes) -> int:
    n = 0
    for b in raw:
        n = (n << 7) | (b & 0x7F)
    return n


def _decode_text_blob(buf: bytes) -> str:
    if not buf:
        return ""
    enc = buf[0]
    raw = buf[1:]
    if enc == 0:
        return raw.split(b"\x00", 1)[0].decode("latin-1", "replace")
    if enc == 1:
        return raw.decode("utf-16", "replace").split("\x00", 1)[0]
    if enc == 2:
        return raw.decode("utf-16-be", "replace").split("\x00", 1)[0]
    if enc == 3:
        return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
    return ""


def _decode_paired(buf: bytes) -> str:
    """Two encoded strings (TXXX description and value)."""
    if not buf:
        return ""
    enc = buf[0]
    raw = buf[1:]
    if enc in (1, 2):
        codec = "utf-16" if enc == 1 else "utf-16-be"
        return " ".join(part for part in raw.decode(codec, "replace").split("\x00") if part)
    codec = "latin-1" if enc == 0 else "utf-8"
    return " ".join(part.decode(codec, "replace") for part in raw.split(b"\x00") if part)


def _decode_comm(buf: bytes) -> str:
    if len(buf) < 5:
        return ""
    enc = buf[0]
    rest = buf[4:]
    if enc == 1:
        text = rest.decode("utf-16", "replace")
        return " ".join(part for part in text.split("\x00") if part)
    if enc == 2:
        text = rest.decode("utf-16-be", "replace")
        return " ".join(part for part in text.split("\x00") if part)
    codec = "latin-1" if enc == 0 else "utf-8"
    return " ".join(part.decode(codec, "replace") for part in rest.split(b"\x00") if part)


def id3_texts(data: bytes) -> list[str]:
    """ID3v2.3/v2.4 text, TXXX, COMM, USLT, plus an ID3v1 tail. No tag rewrite."""
    out: list[str] = []
    if len(data) >= 10 and data[:3] == b"ID3" and data[3] in (3, 4):
        ver = data[3]
        size = _syncsafe(data[6:10])
        pos = 10
        end = min(len(data), 10 + size)
        if data[5] & 0x40 and pos + 4 <= end:
            ext = struct.unpack(">I", data[pos : pos + 4])[0]
            if ver == 4:
                ext = _syncsafe(data[pos : pos + 4])
            pos += max(ext, 4)
        while pos + 10 <= end:
            fid = data[pos : pos + 4]
            if fid == b"\x00\x00\x00\x00" or any(b < 65 or b > 90 for b in fid):
                break
            if ver == 4:
                fsize = _syncsafe(data[pos + 4 : pos + 8])
            else:
                fsize = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
            start = pos + 10
            if fsize < 0 or start > end:
                break
            chunk = data[start : min(end, start + fsize)]
            pos = start + fsize
            if not chunk:
                continue
            if fid in {b"COMM", b"USLT"}:
                text = _decode_comm(chunk)
            elif fid == b"TXXX":
                text = _decode_paired(chunk)
            elif fid.startswith(b"T"):
                text = _decode_text_blob(chunk)
            else:
                continue
            if text:
                out.append(text)
    if len(data) >= 128 and data[-128:-125] == b"TAG":
        tail = data[-128:]
        for raw in (tail[3:33], tail[33:63], tail[63:93], tail[97:127]):
            text = raw.split(b"\x00", 1)[0].decode("latin-1", "replace").strip()
            if text:
                out.append(text)
    return out


def flac_comment_texts(data: bytes) -> list[str]:
    if not data.startswith(b"fLaC"):
        return []
    pos = 4
    out: list[str] = []
    for _ in range(64):
        if pos + 4 > len(data):
            break
        head = data[pos]
        last = bool(head & 0x80)
        kind = head & 0x7F
        ln = int.from_bytes(data[pos + 1 : pos + 4], "big")
        start = pos + 4
        end = start + ln
        if end > len(data):
            break
        block = data[start:end]
        pos = end
        if kind == 4 and len(block) >= 8:
            vlen = struct.unpack_from("<I", block, 0)[0]
            cursor = 4 + vlen
            if cursor + 4 <= len(block):
                count = struct.unpack_from("<I", block, cursor)[0]
                cursor += 4
                for _i in range(min(count, 64)):
                    if cursor + 4 > len(block):
                        break
                    clen = struct.unpack_from("<I", block, cursor)[0]
                    cursor += 4
                    raw = block[cursor : cursor + clen]
                    cursor += clen
                    if raw:
                        out.append(raw.decode("utf-8", "replace"))
        if last or kind == 0 and last:
            break
        if last:
            break
    return out


def _meta_findings(texts: list[str], cre, method: str, evidence: str) -> list[HuntFinding]:
    if not texts:
        return []
    flags: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for flag in find_flags(text, cre):
            if flag not in seen:
                seen.add(flag)
                flags.append(flag)
    if not flags:
        return []
    return [
        HuntFinding(
            family="audio",
            method=method,
            confidence=0.99,
            evidence=evidence,
            text=flags[0][:4096],
            flag_hit=True,
            extra={"flags": flags},
        )
    ]


def decode_compressed_pcm(data: bytes, kind: str) -> tuple[WavPcm | None, str]:
    try:
        import miniaudio
    except ImportError:
        return None, "miniaudio is not installed. PCM walks skipped."
    try:
        if kind == "flac":
            sound = miniaudio.flac_read_s16(data)
        elif kind == "mp3":
            sound = miniaudio.mp3_read_s16(data)
        else:
            return None, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"PCM decode failed ({type(exc).__name__}). Metadata was still read."
    channels = int(sound.nchannels or 0)
    if channels < 1 or channels > 8:
        return None, "Decoded PCM channel count is not usable."
    raw = np.asarray(sound.samples, dtype=np.int16)
    usable = (raw.size // channels) * channels
    if usable < channels:
        return None, "Decoded PCM was empty."
    samples = raw[:usable].reshape(-1, channels).copy()
    from veilscan.hunt.audio import _MAX_SAMPLES

    if samples.shape[0] > _MAX_SAMPLES:
        samples = samples[:_MAX_SAMPLES].copy()
    rate = int(sound.sample_rate or 0)
    if rate <= 0:
        rate = 8000
    return WavPcm(samples=samples, sample_rate=rate, bits=16, channels=channels), ""


def compressed_pass(
    data: bytes,
    kind: str,
    cre,
    *,
    deep: bool = False,
    stop_on_flag: bool = False,
    dest: Path | None = None,
):
    from veilscan.hunt.audio import AudioPass

    passed = AudioPass()
    if kind == "mp3":
        texts = id3_texts(data)
        passed.findings.extend(_meta_findings(texts, cre, "id3", "ID3 text"))
        if texts:
            passed.notes.append(f"ID3 text fields: {len(texts)}.")
    elif kind == "flac":
        texts = flac_comment_texts(data)
        passed.findings.extend(_meta_findings(texts, cre, "flac-comment", "FLAC Vorbis comment"))
        if texts:
            passed.notes.append(f"FLAC Vorbis comments: {len(texts)}.")
    else:
        return passed
    if stop_on_flag and any(f.flag_hit for f in passed.findings):
        return passed
    pcm, note = decode_compressed_pcm(data, kind)
    if note:
        passed.notes.append(note)
    if pcm is None:
        return passed
    if kind == "mp3":
        passed.notes.append("MP3 PCM is the lossy decode. Sample LSB is not preserved by a normal encode.")
    fill_pcm_findings(passed, pcm, cre, deep=deep, stop_on_flag=stop_on_flag, dest=dest)
    return passed


def plant_flac_lsb(flag: str, *, sample_rate: int = 8000) -> bytes:
    """Eval-only. 16-bit sample LSB inside a verbatim FLAC. No plaintext flag."""
    from veilscan.decode.lsb import message_to_bits
    from veilscan.hunt.audio import _walk, _write_low_bits

    payload = message_to_bits(flag, msb_first=True)
    samples = np.zeros((int(payload.size + 64), 1), dtype=np.int16)
    _write_low_bits(_walk(samples, 0, False), payload, 0)
    return encode_verbatim_flac(samples[:, 0], sample_rate)


def plant_flac_comment(flag: str, *, sample_rate: int = 8000) -> bytes:
    """Eval-only. Flag lives in a Vorbis comment on silence."""
    samples = np.zeros(_BLOCK, dtype=np.int16)
    return encode_verbatim_flac(samples, sample_rate, comments=[("DESCRIPTION", flag)])


def plant_mp3_id3(flag: str) -> bytes:
    """Eval-only ID3v2.3 COMM in UTF-16. No MPEG frame and no ASCII flag bytes."""
    encoded = b"\xff\xfe" + flag.encode("utf-16-le")
    # encoding 1, language, empty UTF-16 description, then the value with BOM.
    payload = bytes((1,)) + b"eng" + b"\xff\xfe\x00\x00" + encoded
    frame = b"COMM" + struct.pack(">I", len(payload)) + b"\x00\x00" + payload
    size = len(frame)
    sync = bytes(((size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F))
    return b"ID3" + bytes((3, 0, 0)) + sync + frame
