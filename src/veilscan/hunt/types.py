"""Hunt findings. Separate from DecodeResult.found so photo LSB cannot mint a flag."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

HUNT_SCHEMA = 2


@dataclass
class HuntFinding:
    family: str
    method: str
    confidence: float
    evidence: str
    text: str | None = None
    artifact_name: str | None = None
    offset: int | None = None
    length: int | None = None
    flag_hit: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "family": self.family,
            "method": self.method,
            "confidence": round(float(self.confidence), 6),
            "evidence": self.evidence,
            "flag_hit": bool(self.flag_hit),
        }
        if self.text is not None:
            d["text"] = self.text if len(self.text) <= 4096 else self.text[:4095] + "..."
        if self.artifact_name:
            d["artifact_name"] = self.artifact_name
        if self.offset is not None:
            d["offset"] = int(self.offset)
        if self.length is not None:
            d["length"] = int(self.length)
        if self.extra:
            d["extra"] = self.extra
        return d


@dataclass
class HuntResult:
    schema_version: int = HUNT_SCHEMA
    path: str | None = None
    sha256: str = ""
    kind: str = "unknown"
    size: int = 0
    elapsed_ms: float = 0.0
    findings: list[HuntFinding] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    def status(self) -> str:
        if self.flags:
            return "flags"
        if self.findings:
            return "findings"
        return "empty"

    def summary(self) -> str:
        n_f = len(self.flags)
        n_h = len(self.findings)
        if n_f:
            preview = self.flags[0]
            if n_f == 1:
                return f"1 flag  {preview}"
            return f"{n_f} flags  {preview}"
        if n_h:
            return f"{n_h} findings  no FLAG{{}}"
        return "no flags  no extract hits"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "sha256": self.sha256,
            "kind": self.kind,
            "size": self.size,
            "elapsed_ms": round(float(self.elapsed_ms), 1),
            "status": self.status(),
            "summary": self.summary(),
            "flag_count": len(self.flags),
            "finding_count": len(self.findings),
            "flags": list(self.flags),
            "notes": list(self.notes),
            "artifacts": list(self.artifacts),
            "findings": [f.to_json() for f in self.findings],
        }
