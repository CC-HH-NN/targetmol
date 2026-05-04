"""Generate the final TargetMol text summary."""

from __future__ import annotations

from pathlib import Path

from targetmol.provenance import read_run_metadata


DISCLAIMER = (
    "This report summarizes computational generation, docking, and rule-based screening results for computational hit discovery only. "
    "It does not represent experimental validation."
)


def _build_metadata_lines(metadata: dict[str, object]) -> list[str]:
    """Convert run metadata into summary text."""
    lines: list[str] = []
    status = metadata.get("summary_status")
    if isinstance(status, str):
        lines.append(f"Run status: {status}")
    run_dir = metadata.get("run_dir")
    if isinstance(run_dir, str):
        lines.append(f"Run dir: {run_dir}")
    step_count = metadata.get("step_count")
    step_names = metadata.get("step_names")
    if isinstance(step_count, int) and isinstance(step_names, list) and step_names:
        joined = ", ".join(str(name) for name in step_names)
        lines.append(f"Planned steps ({step_count}): {joined}")
    planned_steps_file = metadata.get("planned_steps_file")
    if isinstance(planned_steps_file, str):
        lines.append(f"Planned steps file: {planned_steps_file}")
    execution_index_file = metadata.get("execution_index_file")
    if isinstance(execution_index_file, str):
        lines.append(f"Execution index: {execution_index_file}")
    command_log = metadata.get("command_log")
    if isinstance(command_log, str):
        lines.append(f"Command log: {command_log}")
    failed_step = metadata.get("failed_step")
    if isinstance(failed_step, str):
        lines.append(f"Failed step: {failed_step}")
    iterative_summary_file = metadata.get("iterative_summary_file")
    if isinstance(iterative_summary_file, str):
        lines.append(f"Iterative summary: {iterative_summary_file}")
    iterative_stop_reason = metadata.get("iterative_stop_reason")
    if isinstance(iterative_stop_reason, str):
        lines.append(f"Iterative stop reason: {iterative_stop_reason}")
    iterative_rounds = metadata.get("iterative_rounds")
    if isinstance(iterative_rounds, int):
        lines.append(f"Iterative rounds: {iterative_rounds}")
    iterative_accepted_updates = metadata.get("iterative_accepted_updates")
    if isinstance(iterative_accepted_updates, int):
        lines.append(f"Iterative accepted updates: {iterative_accepted_updates}")
    iterative_final_candidate_count = metadata.get("iterative_final_candidate_count")
    if isinstance(iterative_final_candidate_count, int):
        lines.append(f"Iterative final candidate count: {iterative_final_candidate_count}")
    iterative_final_smiles = metadata.get("iterative_final_smiles")
    if isinstance(iterative_final_smiles, str):
        lines.append(f"Iterative final candidates: {iterative_final_smiles}")
    return lines


def _extend_unique(lines: list[str], extra_lines: list[str] | None) -> None:
    """Append lines in order while avoiding duplicates."""
    if not extra_lines:
        return
    existing = set(lines)
    for line in extra_lines:
        if not line or line in existing:
            continue
        lines.append(line)
        existing.add(line)


def write_final_summary(*, route: str, final_dir: Path, screening_output: Path | None, extra_lines: list[str] | None = None) -> Path:
    """Write the final concise text summary."""
    final_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_run_metadata(final_dir.parent / "provenance")
    metadata_screening_output = metadata.get("screening_output")
    resolved_screening_output = screening_output
    if resolved_screening_output is None and isinstance(metadata_screening_output, str):
        resolved_screening_output = Path(metadata_screening_output)
    lines = [
        f"Route: {route}",
        f"Screening output: {resolved_screening_output if resolved_screening_output else 'N/A'}",
    ]
    _extend_unique(lines, _build_metadata_lines(metadata))
    _extend_unique(lines, extra_lines)
    lines.append(DISCLAIMER)
    output_path = final_dir / "summary.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
