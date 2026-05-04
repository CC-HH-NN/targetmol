"""Screening data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScreeningCandidate:
    """Unified candidate molecule record."""

    name: str
    smiles: str
    source: str = "input"
    tags: list[str] = field(default_factory=list)


@dataclass
class DockingResult:
    """Docking result for one candidate."""

    name: str
    smiles: str
    docking_score: float
    pose_sdf: str


@dataclass
class FilterResult:
    """Screening-property result for one candidate."""

    name: str
    smiles: str
    is_valid: bool = True
    lipinski_pass: bool = True
    pains_alerts: list[str] = field(default_factory=list)
    sa_score: float | None = None
    metrics: dict[str, float | int | bool | str] = field(default_factory=dict)


@dataclass
class ReportRow:
    """Stable output row in the final report."""

    name: str
    smiles: str
    rank: int
    docking_score: float | None = None
    total_score: float | None = None
    pose_sdf: str | None = None
    source: str = "input"
    tags: list[str] = field(default_factory=list)
