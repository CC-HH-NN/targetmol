"""clean-room screening 数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScreeningCandidate:
    """统一候选分子记录。"""

    name: str
    smiles: str
    source: str = "input"
    tags: list[str] = field(default_factory=list)


@dataclass
class DockingResult:
    """单个候选的对接结果。"""

    name: str
    smiles: str
    docking_score: float
    pose_sdf: str


@dataclass
class FilterResult:
    """单个候选的筛选规则结果。"""

    name: str
    smiles: str
    is_valid: bool = True
    lipinski_pass: bool = True
    pains_alerts: list[str] = field(default_factory=list)
    sa_score: float | None = None
    metrics: dict[str, float | int | bool | str] = field(default_factory=dict)


@dataclass
class ReportRow:
    """最终报告中的稳定输出行。"""

    name: str
    smiles: str
    rank: int
    docking_score: float | None = None
    total_score: float | None = None
    pose_sdf: str | None = None
    source: str = "input"
    tags: list[str] = field(default_factory=list)
