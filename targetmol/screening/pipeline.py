"""clean-room screening 总流程。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from targetmol.screening.docking import run_docking_batch
from targetmol.screening.normalize import load_normalized_smiles_file
from targetmol.screening.properties import evaluate_candidate_properties
from targetmol.screening.ranking import rank_screening_rows
from targetmol.screening.report import write_final_reports


def run_screening_pipeline(
    *,
    normalized_smiles_file: Path,
    receptor_pdb: Path,
    reference_ligand: Path | None,
    output_dir: Path,
    gnina_bin: str,
    top_k: int,
    exhaustiveness: int,
    n_poses: int,
    n_threads: int,
    box_size: tuple[float, float, float],
) -> dict[str, object]:
    """执行 clean-room screening 并写出最终报告。"""
    started_at = perf_counter()
    candidates = load_normalized_smiles_file(normalized_smiles_file)
    candidate_index = {candidate.name: candidate for candidate in candidates}
    docking_rows = run_docking_batch(
        candidates=candidates,
        receptor_pdb=receptor_pdb,
        reference_ligand=reference_ligand,
        output_dir=output_dir,
        gnina_bin=gnina_bin,
        exhaustiveness=exhaustiveness,
        n_poses=n_poses,
        n_threads=n_threads,
        box_size=box_size,
    )

    merged: list[dict[str, object]] = []
    for row in docking_rows:
        candidate_name = str(row["name"])
        candidate = candidate_index[candidate_name]
        merged_row = dict(row)
        merged_row.setdefault("source", candidate.source)
        merged_row.setdefault("tags", candidate.tags)
        merged_row.update(evaluate_candidate_properties(candidate))
        merged.append(merged_row)

    ranked = rank_screening_rows(merged)[:top_k]
    outputs = write_final_reports(
        output_dir,
        ranked,
        metadata={
            "target": receptor_pdb.stem,
            "total_input": len(candidates),
            "valid_ligands": sum(1 for row in merged if row.get("is_valid", False)),
            "docking_success": sum(1 for row in merged if row.get("docking_score") is not None),
            "passed_filters": sum(
                1
                for row in ranked
                if row.get("is_valid", False)
                and row.get("lipinski_pass", False)
                and int(row.get("pains_alert_count", 0) or 0) == 0
            ),
            "execution_time_seconds": round(perf_counter() - started_at, 4),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "count": len(ranked),
        "outputs": outputs,
        "top_candidates": ranked,
    }
