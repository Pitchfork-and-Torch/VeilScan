"""Plugin registry. Call ensure_loaded() before select()."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veilscan.detectors.base import BaseDetector

_REGISTRY: dict[str, BaseDetector] = {}
_LOADED = False


def register(detector: BaseDetector) -> BaseDetector:
    _REGISTRY[detector.name] = detector
    return detector


def get(name: str) -> BaseDetector:
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}. known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_detectors() -> list[BaseDetector]:
    return [d for _, d in sorted(_REGISTRY.items())]


def select(names: Iterable[str] | None = None, tier: str | None = None) -> list[BaseDetector]:
    items = all_detectors()
    if names:
        want = {n.strip() for n in names}
        items = [d for d in items if d.name in want]
    if tier and tier != "all":
        order = ["fast", "frequency", "residual", "deep", "blackbox", "foundation"]
        if tier not in order:
            raise ValueError(f"unknown tier {tier}")
        allowed = set(order[: order.index(tier) + 1])
        # residual is grouped with frequency in the CLI for convenience
        if tier == "frequency":
            allowed.add("residual")
        items = [d for d in items if d.tier in allowed]
    return items


def ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    from veilscan.detectors import register_all

    register_all()
    _LOADED = True
