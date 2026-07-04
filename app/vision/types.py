from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CameraProbeReport:
    backend: str
    ok: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CapturedFrame:
    backend: str
    image_path: Path
    width: int
    height: int
    pixel_format: str
    sensor: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    rgb: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        data.pop("rgb", None)
        return data


@dataclass
class DefectCandidate:
    label: str
    score: float
    summary: str
    box: dict[str, float] | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisionAnalysisResult:
    recognizer_backend: str
    image_path: Path
    candidates: list[DefectCandidate]
    raw_candidates: list[DefectCandidate] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    annotated_image_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recognizer_backend": self.recognizer_backend,
            "image_path": str(self.image_path),
            "annotated_image_path": str(self.annotated_image_path)
            if self.annotated_image_path
            else "",
            "metrics": self.metrics,
            "notes": self.notes,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "raw_candidates": [candidate.to_dict() for candidate in self.raw_candidates],
        }
