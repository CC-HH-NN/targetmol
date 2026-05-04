"""生成 TargetMol 自己的最终智能体报告。"""

from __future__ import annotations

import json
from pathlib import Path


DISCLAIMER = (
    "本报告基于计算生成、对接和规则筛选结果，仅用于 computational hit discovery 参考，"
    "不代表实验验证结论。"
)


def write_agent_report(
    *,
    final_dir: Path,
    route: str,
    target_name: str | None,
    iterative_summary_path: Path | None = None,
    screening_report_path: Path | None = None,
) -> Path:
    """把 clean-room 迭代和筛选结果写成一份可读报告。"""
    final_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "TargetMol Agent Report",
        f"Route: {route}",
        f"Target: {target_name or 'N/A'}",
    ]

    iterative_summary = _read_json_file(iterative_summary_path)
    if iterative_summary:
        lines.extend(_build_iterative_section(iterative_summary))

    screening_report = _read_json_file(screening_report_path)
    if screening_report:
        lines.extend(_build_screening_section(screening_report))

    lines.append(DISCLAIMER)
    output_path = final_dir / "agent_report.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _build_iterative_section(payload: dict[str, object]) -> list[str]:
    """把迭代闭环结果转成可读文本。"""
    lines = ["Iterative Ligand Refinement:"]
    stop_reason = payload.get("stop_reason")
    if isinstance(stop_reason, str):
        lines.append(f"- Stop reason: {stop_reason}")
    rounds = payload.get("rounds")
    if isinstance(rounds, list):
        lines.append(f"- Rounds completed: {len(rounds)}")
    accepted_updates_total = payload.get("accepted_updates_total")
    if isinstance(accepted_updates_total, int):
        lines.append(f"- Accepted updates: {accepted_updates_total}")
    final_candidates = payload.get("final_candidates")
    if isinstance(final_candidates, list):
        lines.append(f"- Final candidate count: {len(final_candidates)}")
    lines.extend(_format_count_map("Dominant issues", payload.get("dominant_issue_counts")))
    lines.extend(_format_count_map("Fallback reasons", payload.get("fallback_counts")))
    lines.extend(_format_count_map("Improvements", payload.get("improvement_counts")))
    return lines


def _build_screening_section(payload: dict[str, object]) -> list[str]:
    """把筛选结果转成可读文本。"""
    lines = ["Screening Result:"]
    for label, key in [
        ("Target", "target"),
        ("Total input", "total_input"),
        ("Valid ligands", "valid_ligands"),
        ("Docking success", "docking_success"),
        ("Passed filters", "passed_filters"),
    ]:
        value = payload.get(key)
        if value is not None:
            lines.append(f"- {label}: {value}")

    top_candidates = payload.get("top_candidates")
    if isinstance(top_candidates, list) and top_candidates:
        lines.append("- Top candidates:")
        for item in top_candidates[:3]:
            if not isinstance(item, dict):
                continue
            rank = item.get("rank", "N/A")
            name = item.get("name", "unknown")
            docking_score = item.get("docking_score", "N/A")
            lines.append(f"  {rank}. {name} (docking: {docking_score})")
    return lines


def _format_count_map(title: str, value: object) -> list[str]:
    """把计数字典转成稳定文本。"""
    if not isinstance(value, dict) or not value:
        return []
    pairs = []
    for key, count in value.items():
        if not isinstance(count, int) or count <= 0:
            continue
        pairs.append(f"{key}={count}")
    if not pairs:
        return []
    return [f"- {title}: {', '.join(pairs)}"]


def _read_json_file(path: Path | None) -> dict[str, object] | None:
    """读取可选 JSON 文件。"""
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
