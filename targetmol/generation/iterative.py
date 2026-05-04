"""把 ligand-based clean-room 候选生成组织成多轮迭代闭环。"""

from __future__ import annotations

import json
from pathlib import Path

from targetmol.generation.molecular_refinement import run_molecular_refinement
from targetmol.generation.refinement import refine_candidate_pool
from targetmol.models import ModelsConfig
from targetmol.target_context.grounding import GroundedTargetContext


def run_iterative_ligand_refinement(
    *,
    models: ModelsConfig,
    grounded_context: GroundedTargetContext | None,
    initial_expansion_payload: dict[str, object],
    run_dir: Path,
    iterations: int,
    shortlist_size: int = 12,
) -> dict[str, object]:
    """围绕扩增候选做多轮 shortlist 与分子修正。"""
    max_rounds = max(1, int(iterations))
    route_iterations_dir = run_dir / "route" / "iterations"
    normalized_iterations_dir = run_dir / "normalized" / "iterations"
    route_iterations_dir.mkdir(parents=True, exist_ok=True)
    normalized_iterations_dir.mkdir(parents=True, exist_ok=True)

    current_payload = initial_expansion_payload
    previous_signature = _candidate_signature(current_payload)
    round_records: list[dict[str, object]] = []
    stop_reason = "max_rounds_reached"
    accepted_updates_total = 0
    dominant_issue_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    improvement_counts: dict[str, int] = {
        "validity_fixed": 0,
        "lipinski_fixed": 0,
        "pains_reduced": 0,
        "sa_improved": 0,
    }

    for round_index in range(1, max_rounds + 1):
        round_name = f"iter_{round_index:02d}"
        round_dir = route_iterations_dir / round_name
        round_dir.mkdir(parents=True, exist_ok=True)

        input_json_path = round_dir / "input_candidates.json"
        input_json_path.write_text(
            json.dumps(current_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        refinement_json_path = round_dir / "candidate_refinement.json"
        refinement_smiles_path = normalized_iterations_dir / f"{round_name}_refined_candidates.smi"
        refinement_payload = refine_candidate_pool(
            models=models,
            grounded_context=grounded_context,
            input_json_path=input_json_path,
            output_json_path=refinement_json_path,
            output_smiles_path=refinement_smiles_path,
            shortlist_size=shortlist_size,
            expansion_payload=current_payload,
        )

        molecular_json_path = round_dir / "molecular_refinement.json"
        molecular_smiles_path = normalized_iterations_dir / f"{round_name}_molecular_refinement_candidates.smi"
        molecular_payload = run_molecular_refinement(
            models=models,
            grounded_context=grounded_context,
            input_json_path=refinement_json_path,
            output_json_path=molecular_json_path,
            output_smiles_path=molecular_smiles_path,
        )

        next_payload = _build_next_round_payload(molecular_payload)
        next_signature = _candidate_signature(next_payload)
        accepted_count = sum(1 for item in molecular_payload.get("records", []) if item.get("accepted"))
        round_dominant_issues = _count_dominant_issues(molecular_payload)
        round_fallback_counts = _count_fallbacks(molecular_payload)
        round_improvements = _count_improvements(molecular_payload)
        accepted_updates_total += accepted_count
        _merge_counts(dominant_issue_counts, round_dominant_issues)
        _merge_counts(fallback_counts, round_fallback_counts)
        _merge_counts(improvement_counts, round_improvements)

        round_records.append(
            {
                "round": round_index,
                "input_json_path": str(input_json_path),
                "refinement_json_path": str(refinement_json_path),
                "molecular_json_path": str(molecular_json_path),
                "refinement_smiles_path": str(refinement_smiles_path),
                "molecular_smiles_path": str(molecular_smiles_path),
                "accepted_updates": accepted_count,
                "candidate_count": len(next_payload.get("candidates", [])),
                "dominant_issue_counts": round_dominant_issues,
                "fallback_counts": round_fallback_counts,
                "improvement_counts": round_improvements,
            }
        )

        current_payload = next_payload
        if not current_payload.get("candidates"):
            stop_reason = "no_candidates"
            break
        if next_signature == previous_signature:
            stop_reason = "no_change"
            break
        previous_signature = next_signature

    final_smiles_path = run_dir / "normalized" / "ligand_agent_candidates.smi"
    _write_candidate_smiles_file(final_smiles_path, current_payload.get("candidates", []))
    _write_top_level_artifacts(run_dir, round_records)

    summary = {
        "stop_reason": stop_reason,
        "rounds": round_records,
        "accepted_updates_total": accepted_updates_total,
        "dominant_issue_counts": dominant_issue_counts,
        "fallback_counts": fallback_counts,
        "improvement_counts": improvement_counts,
        "final_candidates": current_payload.get("candidates", []),
        "final_smiles_path": str(final_smiles_path),
    }
    summary_path = run_dir / "route" / "iterative_ligand_refinement.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _build_next_round_payload(molecular_payload: dict[str, object]) -> dict[str, object]:
    """把单轮分子修正结果转成下一轮候选池。"""
    records = molecular_payload.get("records")
    if not isinstance(records, list):
        return {"candidates": []}
    candidates: list[dict[str, object]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        after = item.get("after")
        before = item.get("before")
        if not isinstance(after, dict) or not isinstance(before, dict):
            continue
        smiles = str(after.get("smiles", "")).strip()
        name = str(after.get("name", "")).strip()
        if not smiles or not name:
            continue
        candidates.append(
            {
                "name": name,
                "smiles": smiles,
                "source": "molecular_refinement",
                "rationale": after.get("rationale") or item.get("fallback_reason"),
                "parent_name": before.get("name"),
                "accepted": bool(item.get("accepted", False)),
            }
        )
    return {"candidates": candidates}


def _candidate_signature(payload: dict[str, object]) -> tuple[tuple[str, str], ...]:
    """为候选池生成稳定签名，用于判断迭代是否还有变化。"""
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        return tuple()
    signature: list[tuple[str, str]] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        smiles = str(item.get("smiles", "")).strip()
        if name and smiles:
            signature.append((name, smiles))
    return tuple(signature)


def _write_candidate_smiles_file(path: Path, candidates: list[object]) -> None:
    """把候选列表统一写成标准 smiles 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in candidates:
            if not isinstance(item, dict):
                continue
            smiles = str(item.get("smiles", "")).strip()
            name = str(item.get("name", "")).strip()
            if smiles and name:
                handle.write(f"{smiles}\t{name}\n")


def _write_top_level_artifacts(run_dir: Path, round_records: list[dict[str, object]]) -> None:
    """把最后一轮的关键结果同步到顶层固定文件名。"""
    if not round_records:
        return
    latest = round_records[-1]
    top_level_targets = {
        Path(latest["refinement_json_path"]): run_dir / "route" / "candidate_refinement.json",
        Path(latest["molecular_json_path"]): run_dir / "route" / "molecular_refinement.json",
        Path(latest["refinement_smiles_path"]): run_dir / "normalized" / "refined_candidates.smi",
        Path(latest["molecular_smiles_path"]): run_dir / "normalized" / "molecular_refinement_candidates.smi",
    }
    for source, target in top_level_targets.items():
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _merge_counts(base: dict[str, int], addition: dict[str, int]) -> None:
    """把一轮统计合并到总计数字典。"""
    for key, value in addition.items():
        if value <= 0:
            continue
        base[key] = base.get(key, 0) + value


def _count_dominant_issues(molecular_payload: dict[str, object]) -> dict[str, int]:
    """统计一轮里各类主弱点出现次数。"""
    counts: dict[str, int] = {}
    for record in _iter_records(molecular_payload):
        issue = str(record.get("dominant_issue", "")).strip()
        if issue:
            counts[issue] = counts.get(issue, 0) + 1
    return counts


def _count_fallbacks(molecular_payload: dict[str, object]) -> dict[str, int]:
    """统计一轮里各种回退原因。"""
    counts: dict[str, int] = {}
    for record in _iter_records(molecular_payload):
        raw_reason = record.get("fallback_reason")
        if raw_reason is None:
            continue
        reason = str(raw_reason).strip()
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _count_improvements(molecular_payload: dict[str, object]) -> dict[str, int]:
    """统计一轮里 before/after 的关键改善次数。"""
    counts = {
        "validity_fixed": 0,
        "lipinski_fixed": 0,
        "pains_reduced": 0,
        "sa_improved": 0,
    }
    for record in _iter_records(molecular_payload):
        before = record.get("before")
        after = record.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        if not bool(before.get("is_valid", True)) and bool(after.get("is_valid", False)):
            counts["validity_fixed"] += 1
        if not bool(before.get("lipinski_pass", False)) and bool(after.get("lipinski_pass", False)):
            counts["lipinski_fixed"] += 1
        if int(after.get("pains_alert_count", 0) or 0) < int(before.get("pains_alert_count", 0) or 0):
            counts["pains_reduced"] += 1
        before_sa = before.get("sa_score")
        after_sa = after.get("sa_score")
        if before_sa is not None and after_sa is not None and float(after_sa) < float(before_sa):
            counts["sa_improved"] += 1
    return counts


def _iter_records(molecular_payload: dict[str, object]) -> list[dict[str, object]]:
    """稳定提取分子修正记录列表。"""
    records = molecular_payload.get("records")
    if not isinstance(records, list):
        return []
    return [item for item in records if isinstance(item, dict)]
