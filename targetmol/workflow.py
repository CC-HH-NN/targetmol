"""TargetMol 工作流输入解析、规划与执行。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from targetmol.adapters.drugflow import (
    build_drugflow_command,
    build_drugflow_runner_command,
    build_sdf_to_smiles_command,
)
from targetmol.inputs import InputSpec
from targetmol.models import TargetMolConfig
from targetmol.pdb_prep import prepare_pdb_inputs_for_run
from targetmol.provenance import ProvenanceRecorder, update_run_metadata
from targetmol.reporting import write_final_summary
from targetmol.router import choose_route
from targetmol.screening_inputs import prepare_screening_input_file
from targetmol.shell import run_command

REDACTED_ENV_VALUE = "<redacted>"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class WorkflowStep:
    """单个工作流步骤。"""

    step: str
    command: list[str]
    cwd: str
    outputs: list[str]
    env: dict[str, str] | None = None


@dataclass
class WorkflowPlan:
    """一次运行的完整步骤计划。"""

    route: str
    run_dir: str
    steps: list[WorkflowStep]


def _sanitize_env_for_provenance(env: dict[str, str] | None) -> dict[str, str]:
    """对 provenance 中的环境变量做脱敏，避免把密钥写入运行目录。"""
    if not env:
        return {}
    sanitized = {}
    for key, value in env.items():
        if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            sanitized[key] = REDACTED_ENV_VALUE
        else:
            sanitized[key] = value
    return sanitized


def _serialize_step_for_provenance(step: WorkflowStep) -> dict[str, object]:
    """把步骤对象转成可安全落盘的结构。"""
    data = asdict(step)
    data["env"] = _sanitize_env_for_provenance(step.env)
    return data


def _pick_screening_output(plan: WorkflowPlan) -> str | None:
    """优先找出供 summary 展示的核心筛选结果路径。"""
    for step in reversed(plan.steps):
        for output in step.outputs:
            path = Path(output)
            if path.name == "final_report.json":
                return output
    for step in reversed(plan.steps):
        for output in step.outputs:
            if "screen" in step.step:
                return output
    for step in reversed(plan.steps):
        if step.outputs:
            return step.outputs[0]
    return None


def _write_run_summary_metadata(
    *,
    plan: WorkflowPlan,
    provenance_dir: Path,
    summary_status: str,
    planned_steps_file: Path | None = None,
    execution_index_file: Path | None = None,
    command_log_file: Path | None = None,
    failed_step: str | None = None,
) -> Path:
    """把 dry-run 和真实执行共用的 summary 上下文写入 provenance。"""
    fields: dict[str, object] = {
        "route": plan.route,
        "run_dir": plan.run_dir,
        "summary_status": summary_status,
        "step_count": len(plan.steps),
        "step_names": [step.step for step in plan.steps],
        "screening_output": _pick_screening_output(plan),
        "command_log": str(command_log_file) if command_log_file and command_log_file.exists() else None,
        "failed_step": failed_step,
    }
    if planned_steps_file is not None:
        fields["planned_steps_file"] = str(planned_steps_file)
    if execution_index_file is not None:
        fields["execution_index_file"] = str(execution_index_file)
    return update_run_metadata(provenance_dir, **fields)


def create_run_dir(base_dir: Path, run_name: str, timestamp: str | None = None) -> Path:
    """创建一次运行对应的目录。"""
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{ts}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ["inputs", "normalized", "route", "screening", "final", "provenance", "logs"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_input_spec_for_run(config: TargetMolConfig, spec: InputSpec, run_dir: Path) -> InputSpec:
    """把运行前需要准备的远程输入解析成本地可执行输入。"""
    if spec.candidate_smiles_file:
        _validate_screen_only_spec(spec)
        resolved = replace(
            spec,
            candidate_smiles_file=prepare_screening_input_file(
                spec.candidate_smiles_file,
                run_dir,
                config.envs.screening_conda_env,
            ),
        )
        if resolved.pdb_id and resolved.pdb_file is None:
            prepared = prepare_pdb_inputs_for_run(resolved.pdb_id, run_dir)
            fallback_target_name = resolved.target_name or prepared.target_name or f"PDB ID {prepared.pdb_id}"
            return replace(
                resolved,
                pdb_id=prepared.pdb_id,
                pdb_file=prepared.pdb_file,
                reference_ligand=resolved.reference_ligand or prepared.reference_ligand,
                target_name=fallback_target_name,
            )
        return resolved
    if spec.pdb_id and spec.pdb_file is None:
        prepared = prepare_pdb_inputs_for_run(spec.pdb_id, run_dir)
        fallback_target_name = spec.target_name or prepared.target_name or f"PDB ID {prepared.pdb_id}"
        return replace(
            spec,
            pdb_id=prepared.pdb_id,
            pdb_file=prepared.pdb_file,
            reference_ligand=spec.reference_ligand or prepared.reference_ligand,
            target_name=fallback_target_name,
        )
    return spec


def plan_workflow(config: TargetMolConfig, spec: InputSpec, run_dir: Path) -> WorkflowPlan:
    """根据输入和配置生成执行计划。"""
    if _is_targetmol_generated_candidate_spec(spec):
        if spec.pdb_file is not None:
            return _plan_targetmol_ligand_screening(config, spec, run_dir)
        return _plan_targetmol_ligand_candidates_only(spec, run_dir)
    if spec.candidate_smiles_file:
        _validate_screen_only_spec(spec)
        return _plan_screen_only(config, spec, run_dir)

    route = choose_route(
        has_pdb=spec.has_pdb,
        has_reference_ligand=spec.has_reference_ligand,
        has_seed_smiles=spec.has_seed_smiles,
        has_target_context=spec.has_target_context,
    )
    if route == "sbdd_drugflow":
        return _plan_sbdd_drugflow(config, spec, run_dir, route)
    if route == "ligand_based_targetmol":
        raise ValueError("ligand_based_targetmol 路线需要先由 CLI 生成候选 SMILES。")
    raise ValueError(f"未知工作流路线: {route}")


def write_plan_snapshot(plan: WorkflowPlan, output_path: Path) -> Path:
    """把工作流计划写入 JSON 文件，供 dry-run 查看。"""
    output_path.write_text(
        json.dumps(
            {
                "route": plan.route,
                "run_dir": plan.run_dir,
                "steps": [_serialize_step_for_provenance(step) for step in plan.steps],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_run_summary_metadata(
        plan=plan,
        provenance_dir=output_path.parent,
        summary_status="planned",
        planned_steps_file=output_path,
    )
    return output_path


def execute_plan(
    plan: WorkflowPlan,
    run_dir: Path,
    provenance: ProvenanceRecorder,
    runner=run_command,
) -> Path:
    """顺序执行工作流步骤，并输出执行索引。"""
    execution_index = []
    logs_dir = run_dir / "logs"
    execution_index_path = run_dir / "provenance" / "execution_index.json"
    for step in plan.steps:
        try:
            result = runner(
                step=step.step,
                command=step.command,
                cwd=Path(step.cwd),
                logs_dir=logs_dir,
                provenance=provenance,
                env=step.env,
            )
        except Exception as exc:
            execution_index.append(
                {
                    "step": step.step,
                    "command": step.command,
                    "cwd": step.cwd,
                    "returncode": None,
                    "outputs": step.outputs,
                    "env": _sanitize_env_for_provenance(step.env),
                    "error": str(exc),
                }
            )
            execution_index_path.write_text(json.dumps(execution_index, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_run_summary_metadata(
                plan=plan,
                provenance_dir=run_dir / "provenance",
                summary_status="failed",
                execution_index_file=execution_index_path,
                command_log_file=provenance.command_log if hasattr(provenance, "command_log") else None,
                failed_step=step.step,
            )
            write_final_summary(
                route=plan.route,
                final_dir=run_dir / "final",
                screening_output=Path(_pick_screening_output(plan)) if _pick_screening_output(plan) else None,
                extra_lines=[
                    f"Failed step: {step.step}",
                    f"Launch error: {exc}",
                ],
            )
            raise RuntimeError(
                "步骤启动失败: "
                f"{step.step}; command={' '.join(step.command)}; error={exc}"
            ) from exc
        execution_index.append(
            {
                "step": step.step,
                "command": step.command,
                "cwd": step.cwd,
                "returncode": result.returncode,
                "outputs": step.outputs,
                "env": _sanitize_env_for_provenance(step.env),
            }
        )
        if result.returncode != 0:
            stdout_log = logs_dir / f"{step.step}.stdout.log"
            stderr_log = logs_dir / f"{step.step}.stderr.log"
            execution_index_path.write_text(json.dumps(execution_index, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_run_summary_metadata(
                plan=plan,
                provenance_dir=run_dir / "provenance",
                summary_status="failed",
                execution_index_file=execution_index_path,
                command_log_file=provenance.command_log if hasattr(provenance, "command_log") else None,
                failed_step=step.step,
            )
            write_final_summary(
                route=plan.route,
                final_dir=run_dir / "final",
                screening_output=Path(_pick_screening_output(plan)) if _pick_screening_output(plan) else None,
                extra_lines=[
                    f"Failed step: {step.step}",
                    f"Stdout log: {stdout_log}",
                    f"Stderr log: {stderr_log}",
                ],
            )
            raise RuntimeError(
                "步骤失败: "
                f"{step.step}; returncode={result.returncode}; "
                f"command={' '.join(step.command)}; "
                f"stdout_log={stdout_log}; stderr_log={stderr_log}"
            )

    execution_index_path.write_text(json.dumps(execution_index, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_run_summary_metadata(
        plan=plan,
        provenance_dir=run_dir / "provenance",
        summary_status="completed",
        execution_index_file=execution_index_path,
        command_log_file=provenance.command_log if hasattr(provenance, "command_log") else None,
    )
    write_final_summary(
        route=plan.route,
        final_dir=run_dir / "final",
        screening_output=Path(_pick_screening_output(plan)) if _pick_screening_output(plan) else None,
        extra_lines=None,
    )
    return execution_index_path


def _plan_screen_only(config: TargetMolConfig, spec: InputSpec, run_dir: Path) -> WorkflowPlan:
    """为已有候选 SMILES 的直接筛选场景生成计划。"""
    screening_dir = run_dir / "screening"
    if spec.pdb_file is None:
        raise ValueError("screen_only 场景当前要求提供可解析到本地 PDB 的输入。")
    command = _build_internal_screening_command(
        conda_env=config.envs.screening_conda_env,
        normalized_smiles_file=spec.candidate_smiles_file,
        receptor_pdb=spec.pdb_file,
        reference_ligand=spec.reference_ligand,
        output_dir=screening_dir,
        config=config,
    )
    return WorkflowPlan(
        route="screen_only",
        run_dir=str(run_dir),
        steps=[
            WorkflowStep(
                step="targetmol_screen",
                command=command,
                cwd=str(PROJECT_ROOT),
                outputs=[
                    str(screening_dir / "final" / "final_report.json"),
                    str(screening_dir / "final" / "final_report.csv"),
                    str(screening_dir / "final" / "ranked_candidates.smi"),
                ],
            )
        ],
    )


def _plan_targetmol_ligand_screening(config: TargetMolConfig, spec: InputSpec, run_dir: Path) -> WorkflowPlan:
    """为 TargetMol 自研 ligand 候选接内部 screening 生成计划。"""
    screening_dir = run_dir / "screening"
    if spec.pdb_file is None or spec.candidate_smiles_file is None:
        raise ValueError("ligand_based_targetmol 筛选场景要求提供本地 PDB 和已生成候选。")
    command = _build_internal_screening_command(
        conda_env=config.envs.screening_conda_env,
        normalized_smiles_file=spec.candidate_smiles_file,
        receptor_pdb=spec.pdb_file,
        reference_ligand=spec.reference_ligand,
        output_dir=screening_dir,
        config=config,
    )
    return WorkflowPlan(
        route="ligand_based_targetmol",
        run_dir=str(run_dir),
        steps=[
            WorkflowStep(
                step="targetmol_screen",
                command=command,
                cwd=str(PROJECT_ROOT),
                outputs=[
                    str(screening_dir / "final" / "final_report.json"),
                    str(screening_dir / "final" / "final_report.csv"),
                    str(screening_dir / "final" / "ranked_candidates.smi"),
                ],
            )
        ],
    )


def _plan_targetmol_ligand_candidates_only(spec: InputSpec, run_dir: Path) -> WorkflowPlan:
    """为只有 clean-room ligand 候选、暂不具备结构筛选条件的场景生成计划。"""
    outputs = []
    if spec.candidate_smiles_file is not None:
        outputs.append(str(spec.candidate_smiles_file))
    return WorkflowPlan(
        route="ligand_based_targetmol",
        run_dir=str(run_dir),
        steps=[],
    )


def _validate_screen_only_spec(spec: InputSpec) -> None:
    """拦截和 screen-only 冲突的生成输入，避免静默走错路线。"""
    conflicting_inputs = []
    if spec.seed_smiles_file is not None:
        conflicting_inputs.append("seed_smiles_file")
    if conflicting_inputs:
        joined = ", ".join(conflicting_inputs)
        raise ValueError(f"已提供候选文件时，不能再同时提供生成相关输入: {joined}")


def _is_targetmol_generated_candidate_spec(spec: InputSpec) -> bool:
    """判断候选文件是否来自 TargetMol 自己的 ligand 生成主链。"""
    return spec.candidate_smiles_file is not None and spec.generated_candidate_source == "targetmol_ligand"


def _plan_sbdd_drugflow(config: TargetMolConfig, spec: InputSpec, run_dir: Path, route: str) -> WorkflowPlan:
    """为 DrugFlow 结构驱动路线生成计划。"""
    if spec.pdb_file is None or spec.reference_ligand is None:
        raise ValueError("sbdd_drugflow 路线要求同时提供本地 PDB 和 reference ligand。")

    generated_sdf = run_dir / "route" / "drugflow_samples.sdf"
    generated_smi = run_dir / "normalized" / "drugflow_samples.smi"
    drugflow_base = build_drugflow_command(
        protein=spec.pdb_file,
        ref_ligand=spec.reference_ligand,
        checkpoint=config.drugflow.checkpoint,
        output=generated_sdf,
        n_samples=config.drugflow.n_samples,
        batch_size=config.drugflow.batch_size,
        n_steps=config.drugflow.n_steps,
        pocket_distance_cutoff=config.drugflow.pocket_distance_cutoff,
        device=config.drugflow.device,
    )
    drugflow_command = build_drugflow_runner_command(
        conda_env=config.envs.drugflow_conda_env,
        root=config.paths.drugflow_root,
        command=drugflow_base,
    )
    smiles_command = build_sdf_to_smiles_command(
        conda_env=config.envs.drugflow_conda_env,
        sdf_path=generated_sdf,
        smiles_path=generated_smi,
    )

    steps = [
        WorkflowStep(
            step="drugflow_generate",
            command=drugflow_command,
            cwd=str(config.paths.drugflow_root),
            outputs=[str(generated_sdf)],
            env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
        ),
        WorkflowStep(
            step="sdf_to_smiles",
            command=smiles_command,
            cwd=str(Path(__file__).resolve().parent.parent),
            outputs=[str(generated_smi)],
        ),
    ]

    screening_dir = run_dir / "screening"
    internal_command = _build_internal_screening_command(
        conda_env=config.envs.screening_conda_env,
        normalized_smiles_file=generated_smi,
        receptor_pdb=spec.pdb_file,
        reference_ligand=spec.reference_ligand,
        output_dir=screening_dir,
        config=config,
    )
    steps.append(
        WorkflowStep(
            step="targetmol_screen",
            command=internal_command,
            cwd=str(PROJECT_ROOT),
            outputs=[
                str(screening_dir / "final" / "final_report.json"),
                str(screening_dir / "final" / "final_report.csv"),
                str(screening_dir / "final" / "ranked_candidates.smi"),
            ],
        )
    )

    return WorkflowPlan(route=route, run_dir=str(run_dir), steps=steps)


def _build_internal_screening_command(
    *,
    conda_env: str,
    normalized_smiles_file: Path,
    receptor_pdb: Path,
    reference_ligand: Path | None,
    output_dir: Path,
    config: TargetMolConfig,
) -> list[str]:
    """构建 clean-room screening runner 命令。"""
    command = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "-m",
        "targetmol.screening.pipeline_runner",
        "--smiles-file",
        str(normalized_smiles_file),
        "--receptor-pdb",
        str(receptor_pdb),
        "--output-dir",
        str(output_dir),
        "--gnina",
        config.tools.gnina,
        "--top-k",
        str(config.screening.top_k),
        "--exhaustiveness",
        str(config.screening.exhaustiveness),
        "--n-poses",
        str(config.screening.n_poses),
        "--n-threads",
        str(config.screening.n_threads),
    ]
    if reference_ligand is not None:
        command.extend(["--reference-ligand", str(reference_ligand)])
    return command
