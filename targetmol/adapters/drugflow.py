"""DrugFlow adapter layer."""

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
    """Build the DrugFlow generate.py command."""
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
    """Wrap a DrugFlow command in conda run."""
    return ["conda", "run", "-n", conda_env, *command]


def build_sdf_to_smiles_command(*, conda_env: str, sdf_path: Path, smiles_path: Path) -> list[str]:
    """Build the helper command that converts DrugFlow SDF output to SMILES."""
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
