"""LSB layout table. JS on the public lamp must copy this order and ids."""

from __future__ import annotations

from typing import TypedDict


class LsbLayout(TypedDict):
    id: str
    walk: str
    bit: int
    msb_first: bool


# Walk: r | g | b | rgb (pixel R,G,B) | rgb-planar (all R, then G, then B).
LSB_LAYOUTS: list[LsbLayout] = [
    {"id": "r-bit0-msb", "walk": "r", "bit": 0, "msb_first": True},
    {"id": "r-bit0-lsb", "walk": "r", "bit": 0, "msb_first": False},
    {"id": "g-bit0-msb", "walk": "g", "bit": 0, "msb_first": True},
    {"id": "g-bit0-lsb", "walk": "g", "bit": 0, "msb_first": False},
    {"id": "b-bit0-msb", "walk": "b", "bit": 0, "msb_first": True},
    {"id": "b-bit0-lsb", "walk": "b", "bit": 0, "msb_first": False},
    {"id": "rgb-bit0-msb", "walk": "rgb", "bit": 0, "msb_first": True},
    {"id": "rgb-bit0-lsb", "walk": "rgb", "bit": 0, "msb_first": False},
    {"id": "rgb-planar-bit0-msb", "walk": "rgb-planar", "bit": 0, "msb_first": True},
    {"id": "rgb-planar-bit0-lsb", "walk": "rgb-planar", "bit": 0, "msb_first": False},
    {"id": "r-bit1-msb", "walk": "r", "bit": 1, "msb_first": True},
    {"id": "r-bit1-lsb", "walk": "r", "bit": 1, "msb_first": False},
]

LAYOUT_BY_ID = {row["id"]: row for row in LSB_LAYOUTS}

ACCEPT_SCORE = 0.85
MAX_MESSAGE_BYTES = 4096
MIN_FRAMED_CHARS = 4
MIN_UNFRAMED_CHARS = 8
DEFAULT_LAYOUT_ID = "r-bit0-msb"


def get_layout(layout_id: str) -> LsbLayout:
    if layout_id not in LAYOUT_BY_ID:
        known = ", ".join(LAYOUT_BY_ID)
        raise KeyError(f"unknown layout {layout_id}. known: {known}")
    return LAYOUT_BY_ID[layout_id]
