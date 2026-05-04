"""DrugFlow 适配层。"""

from __future__ import annotations

from pathlib import Path


def build_drugflow_command(
    *,
    protein: Path,
    ref_ligand: Path,
    checkpoint: Path,
    output: Path,
    n_samples: int,
    batch_size: int,
    n_steps: int,
    pocket_distance_cutoff: float,
    device: str,
) -> list[str]:
    """构建 DrugFlow 官方 generate.py 命令。"""
    return [
        "python",
        "src/generate.py",
        "--protein",
        str(protein),
        "--ref_ligand",
        str(ref_ligand),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--n_samples",
        str(n_samples),
        "--batch_size",
        str(batch_size),
        "--n_steps",
        str(n_steps),
        "--pocket_distance_cutoff",
        str(pocket_distance_cutoff),
        "--device",
        device,
    ]


def build_drugflow_runner_command(*, conda_env: str, root: Path, command: list[str]) -> list[str]:
    """把 DrugFlow 命令包装成 conda run 调用。"""
    return ["conda", "run", "-n", conda_env, *command]


def build_sdf_to_smiles_command(*, conda_env: str, sdf_path: Path, smiles_path: Path) -> list[str]:
    """构建把 DrugFlow SDF 转为 SMILES 的辅助命令。"""
    return [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "-m",
        "targetmol.runtime_rdkit",
        "sdf-to-smiles",
        "--input-sdf",
        str(sdf_path),
        "--output-smi",
        str(smiles_path),
        "--allow-invalid-records",
    ]
