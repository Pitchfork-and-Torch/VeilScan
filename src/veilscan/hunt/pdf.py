"""PDF hunt: metadata, actions, embedded files, invisible text, revisions.

Extracts hidden payloads from an authorized PDF. It does not rewrite the file
and it does not dump ordinary visible page text. Eval plants at the bottom are
gym-only.
"""

from __future__ import annotations

import base64
import io
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from veilscan.hunt.flags import find_flags
from veilscan.hunt.types import HuntFinding

_MAX_OBJECTS = 4000
_MAX_STREAM = 8 * 1024 * 1024
_WS = b"\x00\t\n\x0c\r "
_DELIM = set(_WS + b"()<>[]{}/%")
_OBJ_HEAD = re.compile(rb"(?<![0-9])(\d+)\s+(\d+)\s+obj\b")


class Name(str):
    pass


class PdfStr(str):
    pass


class Ref:
    def __init__(self, num: int, gen: int) -> None:
        self.num = num
        self.gen = gen


@dataclass
class PdfObj:
    num: int
    gen: int
    offset: int
    value: object
    stream: bytes | None = None


@dataclass
class PdfPass:
    findings: list[HuntFinding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    recurse: list[tuple[str, bytes]] = field(default_factory=list)


def pdf_stream_spans(data: bytes) -> list[tuple[int, int]]:
    """Byte ranges of stream payloads so carve does not treat them as files."""
    if not data.startswith(b"%PDF-"):
        return []
    try:
        _objs, spans, _notes = _parse_objects(data)
    except Exception:
        return []
    return spans


def pdf_pass(
    data: bytes,
    cre,
    *,
    dest: Path | None = None,
    stop_on_flag: bool = False,
) -> PdfPass:
    out = PdfPass()
    if not data.startswith(b"%PDF-"):
        out.notes.append("Not a PDF header.")
        return out
    try:
        objs, spans, notes = _parse_objects(data)
    except Exception as exc:
        out.notes.append(f"pdf parse failed: {type(exc).__name__}")
        objs, spans, notes = [], [], []
    out.notes.extend(notes)
    by: dict[int, list[PdfObj]] = {}
    for obj in objs:
        by.setdefault(obj.num, []).append(obj)
    for group in by.values():
        group.sort(key=lambda o: o.offset)

    trailers = _trailer_dicts(data, spans)
    encrypted = any("Encrypt" in t for t in trailers)
    version = _version(data)
    eof_count = data.count(b"%%EOF")
    n_orphan = 0
    n_image = 0
    n_image_skip = 0
    n_embed = 0
    n_js = 0
    n_uri = 0
    n_aa = 0
    n_open = 0
    n_xmp = 0
    seen_text: set[str] = set()

    def _add(
        method: str,
        evidence: str,
        text: str | None,
        *,
        offset: int | None = None,
        confidence: float = 0.9,
        artifact: str | None = None,
    ) -> None:
        hits = find_flags(text or "", cre)
        clip = None
        if text:
            clip = text if len(text) <= 1500 else text[:1499] + "..."
        out.findings.append(
            HuntFinding(
                family="pdf",
                method=method,
                confidence=0.99 if hits else confidence,
                evidence=evidence[:500],
                text=clip,
                offset=offset,
                artifact_name=artifact,
                flag_hit=bool(hits),
                extra={"flags": hits} if hits else {},
            )
        )

    info_num = None
    if trailers:
        info = trailers[-1].get("Info")
        if isinstance(info, Ref):
            info_num = info.num
    if info_num is not None and info_num in by:
        current = by[info_num][-1]
        strings = _dict_strings(current.value)
        if strings:
            _add("info", f"trailer Info object {info_num}", "\n".join(strings), offset=current.offset)

    for num, group in by.items():
        if len(group) < 2:
            continue
        for old in group[:-1]:
            n_orphan += 1
            strings = _dict_strings(old.value)
            decoded = _decoded(old)
            blob_text = decoded.decode("latin-1", "replace") if decoded else ""
            joined = "\n".join(strings)
            probe = joined + "\n" + blob_text
            if not find_flags(probe, cre) and not strings:
                _add(
                    "superseded",
                    f"object {num} at {old.offset} replaced by a later revision ({len(decoded or b'')} stream bytes)",
                    None,
                    offset=old.offset,
                    confidence=0.8,
                )
                continue
            preview = joined or blob_text
            _add(
                "superseded",
                f"object {num} at {old.offset} replaced by a later revision",
                preview,
                offset=old.offset,
            )

    js_seen: set[str] = set()
    uri_seen: set[str] = set()
    for obj in objs:
        def _visit(d, obj=obj):
            nonlocal n_aa, n_open, n_js, n_uri
            if "AA" in d:
                n_aa += 1
            if "OpenAction" in d:
                n_open += 1
            for key in ("JS", "JavaScript"):
                if key not in d:
                    continue
                for text in _resolve_text(d[key], by):
                    if not text or text in js_seen:
                        continue
                    js_seen.add(text)
                    n_js += 1
                    _add("javascript", f"/{key}", text, offset=obj.offset)
            s = d.get("S")
            if isinstance(s, Name) and str(s) == "URI" and isinstance(d.get("URI"), PdfStr):
                uri = str(d["URI"]).strip()
                if uri and uri not in uri_seen:
                    uri_seen.add(uri)
                    n_uri += 1
                    _add("uri", "URI action", uri, offset=obj.offset, confidence=0.75)
            if isinstance(s, Name) and str(s) == "Launch":
                target = _dict_strings(d)
                if target:
                    _add("launch", "Launch action", "\n".join(target), offset=obj.offset, confidence=0.85)

        _walk(obj.value, on_dict=_visit)

    content_ids: set[int] = set()
    for obj in objs:
        _collect_contents(obj.value, content_ids)
    invisible: list[str] = []
    for obj in objs:
        if obj.stream is None or not isinstance(obj.value, dict):
            continue
        sub = obj.value.get("Subtype")
        if isinstance(sub, Name) and str(sub) == "Image":
            continue
        dec = _decoded(obj)
        if not dec:
            continue
        interesting = obj.num in content_ids or b"3 Tr" in dec
        if not interesting:
            continue
        for text in _invisible_strings(dec):
            if text and text not in seen_text:
                seen_text.add(text)
                invisible.append(text)
    for text in invisible:
        _add("invisible-text", "text rendering mode 3", text, confidence=0.97)

    for name, payload, offset in _embedded_payloads(objs):
        n_embed += 1
        art = _write_blob(dest, f"pdf_embed_{_safe(name)}", payload)
        try:
            as_text = payload.decode("utf-8")
        except UnicodeDecodeError:
            as_text = None
        _add(
            "embedded-file",
            name,
            as_text if as_text and len(as_text) <= 4000 else None,
            offset=offset,
            artifact=art,
        )
        if not as_text:
            flags = find_flags(payload.decode("latin-1", "replace"), cre)
            if flags:
                _add("embedded-file", name, "\n".join(flags), offset=offset, artifact=art)
        if _recurse_kind(payload):
            out.recurse.append((f"pdf:embed:{name}", payload))

    for obj in objs:
        if obj.stream is None or not isinstance(obj.value, dict):
            continue
        sub = obj.value.get("Subtype")
        if not (isinstance(sub, Name) and str(sub) == "Image"):
            continue
        n_image += 1
        dec = _decoded(obj)
        if not dec:
            n_image_skip += 1
            continue
        filters = _filter_names(obj.value)
        if "DCTDecode" in filters and dec.startswith(b"\xff\xd8"):
            art = _write_blob(dest, f"pdf_image_{obj.num}.jpg", dec)
            _add("image", f"DCT image object {obj.num}", None, offset=obj.offset, confidence=0.7, artifact=art)
            out.recurse.append((f"pdf:image:{obj.num}", dec))
            continue
        png = _image_png(obj.value, dec, by)
        if png is None:
            n_image_skip += 1
            continue
        art = _write_blob(dest, f"pdf_image_{obj.num}.png", png)
        _add("image", f"image object {obj.num}", None, offset=obj.offset, confidence=0.7, artifact=art)
        out.recurse.append((f"pdf:image:{obj.num}", png))

    for obj in objs:
        if obj.stream is None or not isinstance(obj.value, dict):
            continue
        typ = obj.value.get("Type")
        sub = obj.value.get("Subtype")
        if not (
            (isinstance(typ, Name) and str(typ) == "Metadata")
            or (isinstance(sub, Name) and str(sub) == "XML")
        ):
            continue
        n_xmp += 1
        dec = _decoded(obj)
        if not dec:
            continue
        text = dec.decode("utf-8", "replace")
        if find_flags(text, cre):
            _add("xmp", f"metadata object {obj.num}", text, offset=obj.offset)

    for offset, text in _comments(data, spans):
        if text in seen_text:
            continue
        seen_text.add(text)
        _add("comment", f"comment at {offset}", text, offset=offset, confidence=0.85)

    trail = _trailing(data)
    if trail:
        art = _write_blob(dest, "pdf_trailing.bin", trail)
        try:
            text = trail.decode("utf-8")
        except UnicodeDecodeError:
            text = trail.decode("latin-1", "replace") if len(trail) <= 4000 else None
        _add("trailing", f"{len(trail)} bytes after %%EOF", text, offset=data.rfind(b"%%EOF") + 5, artifact=art)
        if _recurse_kind(trail):
            out.recurse.append(("pdf:trailing", trail))

    if encrypted:
        out.notes.append("PDF /Encrypt is set. Stream decode may be incomplete.")
    if n_image_skip:
        out.notes.append(f"{n_image_skip} image streams were not bit-walked (filter, predictor, or colorspace).")
    if not objs:
        out.notes.append("No PDF objects parsed.")

    evidence = (
        f"PDF-{version} generations={eof_count} objects={len(objs)} "
        f"images={n_image} embed={n_embed} js={n_js} uri={n_uri} "
        f"aa={n_aa} openaction={n_open} xmp={n_xmp} "
        f"superseded={n_orphan} encrypted={int(encrypted)} trail={len(trail)}"
    )
    out.findings.insert(
        0,
        HuntFinding(
            family="pdf",
            method="structure",
            confidence=0.99 if objs else 0.4,
            evidence=evidence,
            flag_hit=False,
        ),
    )
    if stop_on_flag:
        kept: list[HuntFinding] = []
        hit = False
        for f in out.findings:
            kept.append(f)
            if f.flag_hit:
                hit = True
                break
        if hit:
            out.findings = kept
            out.recurse = []
    return out


def _version(data: bytes) -> str:
    m = re.match(rb"%PDF-(\d+\.\d+)", data[:16])
    return m.group(1).decode("ascii") if m else "?"


def _decoded(obj: PdfObj) -> bytes | None:
    if obj.stream is None or not isinstance(obj.value, dict):
        return None
    return _decode_stream(obj.value, obj.stream)


def _parse_objects(data: bytes) -> tuple[list[PdfObj], list[tuple[int, int]], list[str]]:
    notes: list[str] = []
    spans: list[tuple[int, int]] = []
    found: list[PdfObj] = []
    for m in _OBJ_HEAD.finditer(data):
        if _covered(m.start(), spans) or _inside_stream(data, m.start()):
            continue
        num = int(m.group(1))
        gen = int(m.group(2))
        try:
            obj, span = _read_object(data, m.end(), num, gen, m.start())
        except Exception:
            notes.append(f"skip object {num} {gen} at {m.start()}")
            continue
        if obj is None:
            continue
        found.append(obj)
        if span is not None:
            spans.append(span)
        if len(found) >= _MAX_OBJECTS:
            notes.append(f"pdf object cap {_MAX_OBJECTS}")
            break
    return found, spans, notes


def _read_object(
    data: bytes, i: int, num: int, gen: int, offset: int
) -> tuple[PdfObj | None, tuple[int, int] | None]:
    i = _skip(data, i)
    if i >= len(data):
        return None, None
    stream: bytes | None = None
    span = None
    if data.startswith(b"<<", i):
        value, i = _parse_dict(data, i)
        j = _skip(data, i)
        if data.startswith(b"stream", j):
            stream, i, span = _read_stream(data, j, value)
    else:
        value, i = _parse_value(data, i)
    return PdfObj(num, gen, offset, value, stream), span


def _read_stream(
    data: bytes, i: int, dct: dict
) -> tuple[bytes | None, int, tuple[int, int] | None]:
    i += 6  # stream
    if data.startswith(b"\r\n", i):
        i += 2
    elif i < len(data) and data[i] in (10, 13):
        i += 1
    length = _length_of(dct, data)
    if length is None or length < 0 or length > _MAX_STREAM or i + length > len(data):
        end = data.find(b"endstream", i, i + _MAX_STREAM)
        if end < 0:
            return None, i, None
        raw = data[i:end]
        return raw, end + len(b"endstream"), (i, end)
    raw = data[i : i + length]
    j = i + length
    if data.startswith(b"\r\n", j):
        j += 2
    elif j < len(data) and data[j] in (10, 13):
        j += 1
    if data.startswith(b"endstream", j):
        j += 9
    return raw, j, (i, i + length)


def _length_of(dct: dict, data: bytes) -> int | None:
    length = dct.get("Length")
    if isinstance(length, int) and not isinstance(length, bool):
        return length
    if isinstance(length, Ref):
        return _resolve_int(data, length.num, length.gen)
    return None


def _resolve_int(data: bytes, num: int, gen: int) -> int | None:
    pat = re.compile(rb"(?<![0-9])" + str(num).encode("ascii") + rb"\s+" + str(gen).encode("ascii") + rb"\s+obj\b")
    for m in pat.finditer(data):
        if _inside_stream(data, m.start()):
            continue
        i = _skip(data, m.end())
        if i >= len(data) or data.startswith(b"<<", i):
            continue
        try:
            val, j = _parse_number(data, i)
        except Exception:
            continue
        j = _skip(data, j)
        if isinstance(val, int) and not isinstance(val, bool) and data.startswith(b"endobj", j):
            return val
    return None


def _stream_keyword(data: bytes, pos: int) -> int:
    """Last `stream` keyword before pos. Ignore the one inside `endstream`."""
    start = pos
    while True:
        i = data.rfind(b"stream", 0, start)
        if i < 0:
            return -1
        if i >= 3 and data[i - 3 : i] == b"end":
            start = i
            continue
        return i


def _inside_stream(data: bytes, pos: int) -> bool:
    prev_stream = _stream_keyword(data, pos)
    prev_end = data.rfind(b"endstream", 0, pos)
    return prev_stream > prev_end


def _covered(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def _skip(data: bytes, i: int) -> int:
    n = len(data)
    while i < n:
        c = data[i]
        if c in _WS:
            i += 1
            continue
        if c == 0x25:
            i += 1
            while i < n and data[i] not in (10, 13):
                i += 1
            continue
        break
    return i


def _boundary(data: bytes, i: int) -> bool:
    return i >= len(data) or data[i] in _DELIM


def _parse_value(data: bytes, i: int):
    i = _skip(data, i)
    if i >= len(data):
        raise ValueError("eof")
    if data.startswith(b"<<", i):
        return _parse_dict(data, i)
    c = data[i]
    if c == ord("["):
        return _parse_array(data, i)
    if c == ord("("):
        raw, j = _parse_lit(data, i)
        return PdfStr(_decode_text(raw)), j
    if c == ord("<"):
        raw, j = _parse_hex(data, i)
        return PdfStr(_decode_text(raw)), j
    if c == ord("/"):
        name, j = _parse_name(data, i)
        return Name(name), j
    if data.startswith(b"true", i) and _boundary(data, i + 4):
        return True, i + 4
    if data.startswith(b"false", i) and _boundary(data, i + 5):
        return False, i + 5
    if data.startswith(b"null", i) and _boundary(data, i + 4):
        return None, i + 4
    if c in b"+-.0123456789":
        num, j = _parse_number(data, i)
        if isinstance(num, int) and not isinstance(num, bool):
            k = _skip(data, j)
            if k < len(data) and data[k] in b"+-.0123456789":
                num2, k2 = _parse_number(data, k)
                k3 = _skip(data, k2)
                if (
                    isinstance(num2, int)
                    and not isinstance(num2, bool)
                    and data.startswith(b"R", k3)
                    and _boundary(data, k3 + 1)
                ):
                    return Ref(num, num2), k3 + 1
        return num, j
    raise ValueError(f"bad token at {i}")


def _parse_dict(data: bytes, i: int) -> tuple[dict, int]:
    i += 2
    out: dict = {}
    while True:
        i = _skip(data, i)
        if i >= len(data):
            return out, i
        if data.startswith(b">>", i):
            return out, i + 2
        if data[i] != ord("/"):
            return out, i
        key, i = _parse_name(data, i)
        val, i = _parse_value(data, i)
        out[key] = val


def _parse_array(data: bytes, i: int) -> tuple[list, int]:
    i += 1
    out: list = []
    while True:
        i = _skip(data, i)
        if i >= len(data):
            return out, i
        if data[i] == ord("]"):
            return out, i + 1
        val, i = _parse_value(data, i)
        out.append(val)


def _parse_name(data: bytes, i: int) -> tuple[str, int]:
    i += 1
    out = bytearray()
    n = len(data)
    while i < n and data[i] not in _DELIM:
        if data[i] == ord("#") and i + 2 < n:
            try:
                out.append(int(data[i + 1 : i + 3], 16))
                i += 3
                continue
            except ValueError:
                pass
        out.append(data[i])
        i += 1
    return out.decode("latin-1", "replace"), i


def _parse_number(data: bytes, i: int):
    start = i
    if data[i] in b"+-":
        i += 1
    saw_dot = False
    while i < len(data) and (data[i] in b"0123456789" or (data[i] == ord(".") and not saw_dot)):
        if data[i] == ord("."):
            saw_dot = True
        i += 1
    text = data[start:i]
    if not text or text in (b"+", b"-", b"."):
        raise ValueError("number")
    if saw_dot:
        return float(text), i
    return int(text), i


def _parse_lit(data: bytes, i: int) -> tuple[bytes, int]:
    i += 1
    out = bytearray()
    depth = 1
    n = len(data)
    while i < n and depth:
        c = data[i]
        if c == 0x5C:
            if i + 1 >= n:
                break
            e = data[i + 1]
            mapped = {
                ord("n"): 10,
                ord("r"): 13,
                ord("t"): 9,
                ord("b"): 8,
                ord("f"): 12,
                ord("("): 40,
                ord(")"): 41,
                ord("\\"): 92,
            }
            if e in mapped:
                out.append(mapped[e])
                i += 2
                continue
            if e in (10, 13):
                i += 2
                if e == 13 and i < n and data[i] == 10:
                    i += 1
                continue
            if e in b"01234567":
                octv = 0
                k = 0
                i += 1
                while k < 3 and i < n and data[i] in b"01234567":
                    octv = (octv << 3) + (data[i] - 48)
                    i += 1
                    k += 1
                out.append(octv & 255)
                continue
            out.append(e)
            i += 2
            continue
        if c == ord("("):
            depth += 1
            out.append(c)
            i += 1
            continue
        if c == ord(")"):
            depth -= 1
            if depth:
                out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    return bytes(out), i


def _parse_hex(data: bytes, i: int) -> tuple[bytes, int]:
    i += 1
    chars: list[str] = []
    while i < len(data) and data[i] != ord(">"):
        if data[i] not in _WS:
            chars.append(chr(data[i]))
        i += 1
    if i < len(data) and data[i] == ord(">"):
        i += 1
    hs = "".join(chars)
    if len(hs) % 2:
        hs += "0"
    try:
        raw = bytes.fromhex(hs)
    except ValueError:
        raw = b""
    return raw, i


def _decode_text(raw: bytes) -> str:
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", "replace")
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", "replace")
    return raw.decode("latin-1", "replace")


def _filter_names(dct: dict) -> list[str]:
    filt = dct.get("Filter")
    if isinstance(filt, Name):
        return [str(filt)]
    if isinstance(filt, list):
        return [str(x) for x in filt if isinstance(x, Name)]
    return []


def _decode_stream(dct: dict, raw: bytes) -> bytes | None:
    data = raw
    for name in _filter_names(dct):
        try:
            if name == "FlateDecode":
                data = _inflate(data)
                if data is None:
                    return None
            elif name == "ASCIIHexDecode":
                data = bytes.fromhex(re.sub(rb"[^0-9A-Fa-f]", b"", data).decode("ascii") or "")
            elif name == "ASCII85Decode":
                data = base64.a85decode(data, adobe=True)
            elif name in {"DCTDecode", "JPXDecode"}:
                continue
            else:
                return None
        except Exception:
            return None
    return data


def _inflate(data: bytes) -> bytes | None:
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            continue
    return None


def _latest(by: dict[int, list[PdfObj]], num: int) -> PdfObj | None:
    group = by.get(num)
    return group[-1] if group else None


def _resolve_text(value, by: dict[int, list[PdfObj]], depth: int = 0) -> list[str]:
    if depth > 4:
        return []
    if isinstance(value, PdfStr):
        return [str(value)]
    if isinstance(value, Ref):
        obj = _latest(by, value.num)
        if obj is None:
            return []
        if obj.stream is not None and isinstance(obj.value, dict):
            dec = _decode_stream(obj.value, obj.stream)
            if not dec:
                return []
            return [dec.decode("latin-1", "replace")]
        return _resolve_text(obj.value, by, depth + 1)
    return []


def _dict_strings(value, limit: int = 24) -> list[str]:
    out: list[str] = []

    def rec(v, depth: int) -> None:
        if len(out) >= limit or depth > 5:
            return
        if isinstance(v, PdfStr):
            text = str(v).strip()
            if text:
                out.append(text[:800])
        elif isinstance(v, dict):
            for item in v.values():
                rec(item, depth + 1)
        elif isinstance(v, list):
            for item in v:
                rec(item, depth + 1)

    rec(value, 0)
    return out


def _walk(value, on_dict) -> None:
    if isinstance(value, dict):
        on_dict(value)
        for item in value.values():
            _walk(item, on_dict)
    elif isinstance(value, list):
        for item in value:
            _walk(item, on_dict)


def _collect_contents(value, ids: set[int]) -> None:
    if isinstance(value, dict):
        if "Contents" in value:
            _add_refs(value["Contents"], ids)
        for item in value.values():
            _collect_contents(item, ids)
    elif isinstance(value, list):
        for item in value:
            _collect_contents(item, ids)


def _add_refs(value, ids: set[int]) -> None:
    if isinstance(value, Ref):
        ids.add(value.num)
    elif isinstance(value, list):
        for item in value:
            _add_refs(item, ids)


def _invisible_strings(data: bytes) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(data)
    while i < n:
        b = data.find(b"BT", i)
        if b < 0:
            break
        if not _tok_edge(data, b, 2):
            i = b + 2
            continue
        e = data.find(b"ET", b + 2)
        if e < 0 or not _tok_edge(data, e, 2):
            break
        try:
            out.extend(_tr3_text(data[b + 2 : e]))
        except Exception:
            pass
        i = e + 2
    return out


def _tok_edge(data: bytes, i: int, length: int) -> bool:
    before = i == 0 or data[i - 1] in _DELIM
    after = i + length >= len(data) or data[i + length] in _DELIM
    return before and after


def _tr3_text(chunk: bytes) -> list[str]:
    mode = 0
    i = 0
    found: list[str] = []
    n = len(chunk)
    while i < n:
        c = chunk[i]
        if c in _WS:
            i += 1
            continue
        if c == ord("("):
            raw, j = _parse_lit(chunk, i)
            text = _decode_text(raw)
            k = _skip(chunk, j)
            if mode == 3 and chunk.startswith(b"Tj", k):
                if text:
                    found.append(text)
            i = j
            continue
        if c == ord("<") and not chunk.startswith(b"<<", i):
            raw, j = _parse_hex(chunk, i)
            text = _decode_text(raw)
            k = _skip(chunk, j)
            if mode == 3 and chunk.startswith(b"Tj", k):
                if text:
                    found.append(text)
            i = j
            continue
        if c == ord("["):
            try:
                arr, j = _parse_array(chunk, i)
            except Exception:
                i += 1
                continue
            k = _skip(chunk, j)
            if mode == 3 and chunk.startswith(b"TJ", k):
                parts = _dict_strings(arr)
                if parts:
                    found.append("".join(parts))
            i = j
            continue
        if c in b"+-.0123456789":
            try:
                num, j = _parse_number(chunk, i)
            except Exception:
                i += 1
                continue
            k = _skip(chunk, j)
            if isinstance(num, int) and not isinstance(num, bool) and chunk.startswith(b"Tr", k):
                mode = num
                i = k + 2
                continue
            i = j
            continue
        i += 1
    return found


def _embedded_payloads(objs: list[PdfObj]) -> list[tuple[str, bytes, int]]:
    names: dict[int, str] = {}
    for obj in objs:
        if not isinstance(obj.value, dict):
            continue
        ef = obj.value.get("EF")
        if not isinstance(ef, dict):
            continue
        fname = f"obj-{obj.num}"
        label = obj.value.get("UF") or obj.value.get("F")
        if isinstance(label, PdfStr) and str(label).strip():
            fname = str(label).strip()
        ref = ef.get("UF") or ef.get("F")
        if isinstance(ref, Ref):
            names[ref.num] = fname
    out: list[tuple[str, bytes, int]] = []
    seen: set[int] = set()
    for obj in objs:
        if obj.stream is None or not isinstance(obj.value, dict):
            continue
        typ = obj.value.get("Type")
        is_emb = (isinstance(typ, Name) and str(typ) == "EmbeddedFile") or obj.num in names
        if not is_emb or obj.num in seen:
            continue
        seen.add(obj.num)
        dec = _decode_stream(obj.value, obj.stream)
        if dec is None:
            continue
        out.append((names.get(obj.num, f"obj-{obj.num}"), dec, obj.offset))
    return out


def _image_png(dct: dict, decoded: bytes, by: dict[int, list[PdfObj]]) -> bytes | None:
    w = _as_int(dct.get("Width"))
    h = _as_int(dct.get("Height"))
    bpc = _as_int(dct.get("BitsPerComponent")) or 8
    if w is None or h is None or w <= 0 or h <= 0 or w > 8000 or h > 8000 or bpc != 8:
        return None
    if w * h > 1_500_000:
        return None
    ncomp = _ncomp(_resolve_value(dct.get("ColorSpace"), by))
    if ncomp not in (1, 3):
        return None
    raw = _unpredict(dct, decoded, w, h, ncomp)
    if raw is None or len(raw) < w * h * ncomp:
        return None
    raw = raw[: w * h * ncomp]
    from PIL import Image

    mode = "L" if ncomp == 1 else "RGB"
    im = Image.frombytes(mode, (w, h), raw)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _resolve_value(value, by: dict[int, list[PdfObj]], depth: int = 0):
    if isinstance(value, Ref) and depth < 4:
        obj = _latest(by, value.num)
        if obj is None:
            return value
        return _resolve_value(obj.value, by, depth + 1)
    return value


def _ncomp(cs) -> int | None:
    if isinstance(cs, Name):
        if str(cs) == "DeviceGray":
            return 1
        if str(cs) == "DeviceRGB":
            return 3
        return None
    if isinstance(cs, list) and cs:
        head = cs[0]
        if isinstance(head, Name) and str(head) in {"DeviceRGB", "CalRGB"}:
            return 3
        if isinstance(head, Name) and str(head) in {"DeviceGray", "CalGray"}:
            return 1
    return None


def _unpredict(dct: dict, data: bytes, w: int, h: int, ncomp: int) -> bytes | None:
    params = dct.get("DecodeParms") or dct.get("DP")
    if isinstance(params, list) and params:
        params = params[0]
    if isinstance(params, Ref):
        params = None
    predictor = 1
    if isinstance(params, dict):
        pred = _as_int(params.get("Predictor"))
        if pred is not None:
            predictor = pred
    columns = w * ncomp
    if predictor <= 1:
        return data
    if predictor >= 10:
        return _unfilter_rows(data, columns)
    return None


def _unfilter_rows(data: bytes, columns: int) -> bytes | None:
    if columns <= 0:
        return None
    stride = columns + 1
    if len(data) % stride != 0:
        return None
    out = bytearray()
    prev = bytearray(columns)
    rows = len(data) // stride
    for r in range(rows):
        filt = data[r * stride]
        row = bytearray(data[r * stride + 1 : (r + 1) * stride])
        if filt == 0:
            pass
        elif filt == 1:
            for i in range(columns):
                left = row[i - 1] if i else 0
                row[i] = (row[i] + left) & 255
        elif filt == 2:
            for i in range(columns):
                row[i] = (row[i] + prev[i]) & 255
        elif filt == 3:
            for i in range(columns):
                left = row[i - 1] if i else 0
                row[i] = (row[i] + ((left + prev[i]) // 2)) & 255
        elif filt == 4:
            for i in range(columns):
                left = row[i - 1] if i else 0
                up = prev[i]
                ul = prev[i - 1] if i else 0
                row[i] = (row[i] + _paeth(left, up, ul)) & 255
        else:
            return None
        out += row
        prev = row
    return bytes(out)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _as_int(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _comments(data: bytes, spans: list[tuple[int, int]]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    i = 0
    n = len(data)
    while i < n:
        if data[i] == 0x25 and not _covered(i, spans):
            j = i + 1
            while j < n and data[j] not in (10, 13):
                j += 1
            raw = data[i + 1 : j]
            start = i
            i = j + 1
            if raw.startswith(b"PDF-") or raw.startswith(b"%"):
                continue
            if any(b > 126 or b < 32 for b in raw if b not in (9,)):
                continue
            text = raw.decode("latin-1").strip()
            if len(text) >= 8:
                out.append((start, text))
            continue
        i += 1
    return out


def _trailing(data: bytes) -> bytes:
    i = data.rfind(b"%%EOF")
    if i < 0:
        return b""
    rest = data[i + 5 :]
    if rest.startswith(b"\r\n"):
        rest = rest[2:]
    elif rest[:1] in (b"\n", b"\r"):
        rest = rest[1:]
    if not rest.strip(b"\x00\t\r\n "):
        return b""
    return rest


def _trailer_dicts(data: bytes, spans: list[tuple[int, int]]) -> list[dict]:
    out: list[dict] = []
    start = 0
    while True:
        i = data.find(b"trailer", start)
        if i < 0:
            break
        start = i + 7
        if _covered(i, spans):
            continue
        if i > 0 and data[i - 1] not in _WS:
            continue
        j = _skip(data, i + 7)
        if not data.startswith(b"<<", j):
            continue
        try:
            dct, _end = _parse_dict(data, j)
        except Exception:
            continue
        out.append(dct)
    return out


def _recurse_kind(blob: bytes) -> bool:
    if len(blob) < 24:
        return False
    if blob.startswith(b"%PDF-"):
        return b"%%EOF" in blob
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if blob.startswith(b"\xff\xd8"):
        return True
    if blob.startswith(b"GIF87a") or blob.startswith(b"GIF89a"):
        return True
    if blob.startswith(b"BM"):
        return True
    if blob.startswith(b"RIFF") and len(blob) >= 12 and blob[8:12] in (b"WAVE", b"WEBP"):
        return True
    if blob[:4] in (b"II*\x00", b"MM\x00*") or blob[:4] == b"II+\x00":
        return True
    return False


def _safe(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in name)
    return (cleaned[:60] or "file")


def _write_blob(dest: Path | None, name: str, payload: bytes) -> str | None:
    if dest is None:
        return None
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / name
    path.write_bytes(payload[:_MAX_STREAM])
    return name


# --- eval plants (gym only) -------------------------------------------------


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _stream_body(dict_body: str, payload: bytes) -> bytes:
    head = f"<< {dict_body} /Length {len(payload)} >>\nstream\n".encode("ascii")
    return head + payload + b"\nendstream"


def assemble_pdf(
    objects: list[bytes],
    *,
    root: int = 1,
    info: int | None = 5,
    comment: str | None = None,
    tail: bytes = b"",
) -> tuple[bytes, int]:
    out = bytearray(b"%PDF-1.4\n")
    if comment:
        line = comment.encode("ascii")
        if not line.startswith(b"%"):
            line = b"% " + line
        out += line + b"\n"
    out += b"%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for body in objects:
        offsets.append(len(out))
        out += f"{len(offsets) - 1} 0 obj\n".encode("ascii")
        out += body
        if not body.endswith(b"\n"):
            out += b"\n"
        out += b"endobj\n"
    xref = len(out)
    size = len(objects) + 1
    out += f"xref\n0 {size}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        line = f"{off:010d} 00000 n \n".encode("ascii")
        if len(line) != 20:
            raise RuntimeError("xref line width")
        out += line
    info_part = f" /Info {info} 0 R" if info else ""
    out += f"trailer\n<< /Size {size} /Root {root} 0 R{info_part} >>\n".encode("ascii")
    out += f"startxref\n{xref}\n%%EOF\n".encode("ascii")
    out += tail
    return bytes(out), xref


def _page_objects(content: bytes, info: bytes, catalog_extra: str = "") -> list[bytes]:
    cat = f"<< /Type /Catalog /Pages 2 0 R {catalog_extra} >>".encode("ascii")
    pages = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    page = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << >> >>"
    return [cat, pages, page, content, info]


def plant_pdf_info(flag: str) -> bytes:
    info = f"<< /Title (Note) /Keywords ({_esc(flag)}) >>".encode("ascii")
    content = _stream_body("", b"BT (Hello) Tj ET")
    blob, _xref = assemble_pdf(_page_objects(content, info))
    return blob


def plant_pdf_js(flag: str) -> bytes:
    script = zlib.compress(f"app.alert('{flag}');".encode("ascii"))
    content = _stream_body("", b"BT (Hello) Tj ET")
    info = b"<< /Title (Note) >>"
    action = b"<< /S /JavaScript /JS 7 0 R >>"
    js = _stream_body("/Filter /FlateDecode", script)
    blob, _xref = assemble_pdf(
        _page_objects(content, info, "/OpenAction 6 0 R") + [action, js],
        info=5,
    )
    return blob


def plant_pdf_embed(flag: str) -> bytes:
    payload = zlib.compress((flag + "\n").encode("ascii"))
    content = _stream_body("", b"BT (Hello) Tj ET")
    info = b"<< /Title (Note) >>"
    spec = b"<< /Type /Filespec /F (flag.txt) /EF << /F 7 0 R >> >>"
    embedded = _stream_body("/Type /EmbeddedFile /Filter /FlateDecode", payload)
    extra = "/Names << /EmbeddedFiles << /Names [(flag.txt) 6 0 R] >> >>"
    blob, _xref = assemble_pdf(_page_objects(content, info, extra) + [spec, embedded], info=5)
    return blob


def plant_pdf_trail(flag: str) -> bytes:
    info = b"<< /Title (Note) >>"
    content = _stream_body("", b"BT (Hello) Tj ET")
    blob, _xref = assemble_pdf(_page_objects(content, info), tail=b"\n" + flag.encode("ascii") + b"\n")
    return blob


def plant_pdf_comment(flag: str) -> bytes:
    info = b"<< /Title (Note) >>"
    content = _stream_body("", b"BT (Hello) Tj ET")
    blob, _xref = assemble_pdf(_page_objects(content, info), comment=flag)
    return blob


def plant_pdf_invisible(flag: str) -> bytes:
    page = f"BT\n3 Tr\n({_esc(flag)}) Tj\nET\n".encode("ascii")
    content = _stream_body("/Filter /FlateDecode", zlib.compress(page))
    info = b"<< /Title (Note) >>"
    blob, _xref = assemble_pdf(_page_objects(content, info))
    return blob


def plant_pdf_incremental(flag: str) -> bytes:
    info = f"<< /Keywords ({_esc(flag)}) >>".encode("ascii")
    content = _stream_body("", b"BT (Hello) Tj ET")
    base, xref = assemble_pdf(_page_objects(content, info), info=5)
    return _append_revision(base, 5, b"<< /Keywords (visible) >>", root=1, info=5, prev=xref)


def plant_pdf_nested_png(png: bytes) -> bytes:
    content = _stream_body("", b"BT (Hello) Tj ET")
    info = b"<< /Title (Note) >>"
    spec = b"<< /Type /Filespec /F (hidden.png) /EF << /F 7 0 R >> >>"
    embedded = _stream_body("/Type /EmbeddedFile", png)
    extra = "/Names << /EmbeddedFiles << /Names [(hidden.png) 6 0 R] >> >>"
    blob, _xref = assemble_pdf(_page_objects(content, info, extra) + [spec, embedded], info=5)
    return blob


def _append_revision(base: bytes, obj_num: int, new_body: bytes, *, root: int, info: int, prev: int) -> bytes:
    out = bytearray(base)
    if not out.endswith(b"\n"):
        out += b"\n"
    off = len(out)
    out += f"{obj_num} 0 obj\n".encode("ascii")
    out += new_body
    if not new_body.endswith(b"\n"):
        out += b"\n"
    out += b"endobj\n"
    xref = len(out)
    out += f"xref\n{obj_num} 1\n".encode("ascii")
    line = f"{off:010d} 00000 n \n".encode("ascii")
    if len(line) != 20:
        raise RuntimeError("xref line width")
    out += line
    size = obj_num + 1
    out += (
        f"trailer\n<< /Size {size} /Root {root} 0 R /Info {info} 0 R /Prev {prev} >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)
