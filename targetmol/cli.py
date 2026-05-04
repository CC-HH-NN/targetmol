"""TargetMol unified command-line entry point."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from targetmol.agent_report import write_agent_report
from targetmol.config import load_config
from targetmol.generation.expansion import expand_candidate_pool
from targetmol.generation.iterative import run_iterative_ligand_refinement
from targetmol.inputs import build_input_spec
from targetmol.provenance import ProvenanceRecorder
from targetmol.provenance import update_run_metadata
from targetmol.request_understanding import enrich_input_spec_from_request
from targetmol.reporting import write_final_summary
from targetmol.target_context.grounding import ground_input_spec_with_context_data
from targetmol.workflow import (
    create_run_dir,
    execute_plan,
    plan_workflow,
    resolve_input_spec_for_run,
    write_plan_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="TargetMol unified CLI")
    parser.add_argument("--config", default="targetmol.yaml")
    parser.add_argument("--request")
    parser.add_argument("--pdb-id")
    parser.add_argument("--pdb-file")
    parser.add_argument("--reference-ligand")
    parser.add_argument("--seed-smiles-file")
    parser.add_argument("--candidate-smiles-file")
    parser.add_argument("--candidate-file")
    parser.add_argument("--target-name")
    parser.add_argument("--disease")
    parser.add_argument("--run-name", default="targetmol_run")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    """Run the CLI entry point."""
    parser = build_parser()
    try:
        args = parser.parse_args()
        config = load_config(args.config)
        spec = enrich_input_spec_from_request(build_input_spec(args), config.models)
        run_dir = create_run_dir(Path(config.project.runs_dir), spec.run_name)
        spec, grounded_context = ground_input_spec_with_context_data(
            spec,
            config.models,
            serper_api_key=config.search.serper_api_key,
            output_path=run_dir / "route" / "seed_grounding.json",
        )
        resolved_spec = resolve_input_spec_for_run(config, spec, run_dir)
        iterative_summary = _maybe_run_ligand_generation(
            config=config,
            spec=resolved_spec,
            grounded_context=grounded_context,
            run_dir=run_dir,
        )
        resolved_spec = _attach_iterative_candidates(resolved_spec, iterative_summary)
        plan = plan_workflow(config, resolved_spec, run_dir)
        plan_path = write_plan_snapshot(plan, run_dir / "provenance" / "planned_steps.json")
        summary_path = write_final_summary(
            route=plan.route,
            final_dir=run_dir / "final",
            screening_output=None,
            extra_lines=[f"Planned steps file: {plan_path}"],
        )
        if args.dry_run:
            print(f"TargetMol route: {plan.route}")
            print(f"Run dir: {run_dir}")
            print(f"Plan file: {plan_path}")
            print(f"Summary file: {summary_path}")
            return 0
        provenance = ProvenanceRecorder(run_dir / "provenance")
        execution_index = execute_plan(plan, run_dir, provenance)
        screening_report_path = run_dir / "screening" / "final" / "final_report.json"
        iterative_summary_path = run_dir / "route" / "iterative_ligand_refinement.json"
        write_agent_report(
            final_dir=run_dir / "final",
            route=plan.route,
            target_name=getattr(resolved_spec, "target_name", None),
            iterative_summary_path=iterative_summary_path if iterative_summary_path.exists() else None,
            screening_report_path=screening_report_path if screening_report_path.exists() else None,
        )
        print(f"TargetMol route: {plan.route}")
        print(f"Run dir: {run_dir}")
        print(f"Execution index: {execution_index}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"TargetMol error: {exc}", file=sys.stderr)
        return 1


def _should_run_ligand_generation(spec, grounded_context) -> bool:
    """Check whether ligand generation should run."""
    if getattr(spec, "candidate_smiles_file", None) is not None:
        return False
    if getattr(spec, "seed_smiles_file", None) is not None:
        return True
    if grounded_context is None:
        return False
    has_pdb = getattr(spec, "pdb_file", None) is not None
    has_reference_ligand = getattr(spec, "reference_ligand", None) is not None
    if has_pdb and has_reference_ligand:
        return False
    return True


def _maybe_run_ligand_generation(
    *,
    config,
    spec,
    grounded_context,
    run_dir: Path,
) -> dict[str, object] | None:
    """Run ligand generation when needed."""
    if not _should_run_ligand_generation(spec, grounded_context):
        return None

    expansion_payload = expand_candidate_pool(
        models=config.models,
        grounded_context=grounded_context or None,
        seed_smiles_file=spec.seed_smiles_file,
        output_json_path=run_dir / "route" / "candidate_expansion.json",
        output_smiles_path=run_dir / "normalized" / "candidate_expansion.smi",
    )
    if not isinstance(expansion_payload.get("candidates"), list):
        return None
    if not expansion_payload["candidates"]:
        return _write_empty_iterative_summary(
            run_dir=run_dir,
            final_smiles_path=run_dir / "normalized" / "candidate_expansion.smi",
            stop_reason=str(expansion_payload.get("degraded_reason") or "no_candidates_from_expansion"),
        )

    iterative_summary = run_iterative_ligand_refinement(
        models=config.models,
        grounded_context=grounded_context or None,
        initial_expansion_payload=expansion_payload,
        run_dir=run_dir,
        iterations=config.ligand_generation.iterations,
    )
    update_run_metadata(
        run_dir / "provenance",
        iterative_summary_file=str(run_dir / "route" / "iterative_ligand_refinement.json"),
        iterative_stop_reason=iterative_summary.get("stop_reason"),
        iterative_rounds=len(iterative_summary.get("rounds", [])),
        iterative_accepted_updates=iterative_summary.get("accepted_updates_total"),
        iterative_final_candidate_count=len(iterative_summary.get("final_candidates", [])),
        iterative_final_smiles=iterative_summary.get("final_smiles_path"),
    )
    return iterative_summary


def _write_empty_iterative_summary(*, run_dir: Path, final_smiles_path: Path, stop_reason: str) -> dict[str, object]:
    """Write an empty candidate summary after unsuccessful expansion."""
    summary = {
        "stop_reason": stop_reason,
        "rounds": [],
        "accepted_updates_total": 0,
        "dominant_issue_counts": {},
        "fallback_counts": {},
        "improvement_counts": {
            "validity_fixed": 0,
            "lipinski_fixed": 0,
            "pains_reduced": 0,
            "sa_improved": 0,
        },
        "final_candidates": [],
        "final_smiles_path": str(final_smiles_path),
    }
    summary_path = run_dir / "route" / "iterative_ligand_refinement.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_run_metadata(
        run_dir / "provenance",
        iterative_summary_file=str(summary_path),
        iterative_stop_reason=summary["stop_reason"],
        iterative_rounds=0,
        iterative_accepted_updates=0,
        iterative_final_candidate_count=0,
        iterative_final_smiles=str(final_smiles_path),
    )
    return summary


def _attach_iterative_candidates(spec, iterative_summary: dict[str, object] | None):
    """Attach iterative generation results as downstream workflow candidates."""
    if iterative_summary is None or getattr(spec, "candidate_smiles_file", None) is not None:
        return spec
    if getattr(spec, "pdb_file", None) is not None and getattr(spec, "reference_ligand", None) is not None:
        return spec

    final_smiles_path = iterative_summary.get("final_smiles_path")
    if not final_smiles_path:
        return spec
    return replace(
        spec,
        candidate_smiles_file=Path(str(final_smiles_path)),
        generated_candidate_source="targetmol_ligand",
    )


if __name__ == "__main__":
    raise SystemExit(main())
