"""Normalize screening candidate files before workflow execution."""

from __future__ import annotations

import subprocess
from pathlib import Path

SMILES_SUFFIXES = {".smi", ".smiles", ".txt"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_screening_input_file(candidate_file: Path, run_dir: Path, conda_env: str) -> Path:
    """Normalize candidate SMILES or SDF files into a .smi file."""
    source = candidate_file.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Candidate molecule file does not exist: {source}")

    normalized_dir = run_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    output_file = normalized_dir / f"{source.stem}.smi"
    suffix = source.suffix.lower()

    if suffix in SMILES_SUFFIXES:
        return _normalize_smiles_like_file(source, output_file, conda_env)
    if suffix == ".sdf":
        return _convert_sdf_to_smiles(source, output_file, conda_env)

    raise ValueError("Unsupported candidate file suffix; supported suffixes are .smi, .smiles, .txt, and .sdf.")


def _normalize_smiles_like_file(source: Path, output_file: Path, conda_env: str) -> Path:
    """Normalize a text candidate file into smiles-tab-name records."""
    command = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "-m",
        "targetmol.runtime_rdkit",
        "normalize-smiles-file",
        "--input-file",
        str(source),
        "--output-smi",
        str(output_file),
        "--allow-invalid-records",
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    _write_normalization_log(output_file, command, result)
    if result.returncode != 0:
        raise RuntimeError(
            "SMILES text normalization failed: "
            f"{source}; command={' '.join(command)}; "
            f"stdout={result.stdout.strip()}; stderr={result.stderr.strip()}"
        )
    if not output_file.exists():
        raise RuntimeError(
            "SMILES text normalization failed：No output file was generated."
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    output_text = output_file.read_text(encoding="utf-8")
    if not output_text.strip():
        raise RuntimeError(
            "SMILES text normalization failed：No usable molecules were generated."
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    if not _has_valid_smiles_record(output_text):
        raise RuntimeError(
            "SMILES text normalization failed：Output content is not a valid SMILES record."
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    return output_file


def _convert_sdf_to_smiles(source: Path, output_file: Path, conda_env: str) -> Path:
    """Convert an SDF candidate file into a standard .smi file."""
    command = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        "-m",
        "targetmol.runtime_rdkit",
        "sdf-to-smiles",
        "--input-sdf",
        str(source),
        "--output-smi",
        str(output_file),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    _write_normalization_log(output_file, command, result)
    if result.returncode != 0:
        raise RuntimeError(
            "SDF to SMILES conversion failed: "
            f"{source}; command={' '.join(command)}; "
            f"stdout={result.stdout.strip()}; stderr={result.stderr.strip()}"
        )
    if not output_file.exists():
        raise RuntimeError(
            "SDF to SMILES conversion failed：No output file was generated."
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    output_text = output_file.read_text(encoding="utf-8")
    if not output_text.strip():
        raise RuntimeError(
            "SDF to SMILES conversion failed：No usable molecules were generated."
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    if not _has_valid_smiles_record(output_text):
        raise RuntimeError(
            "SDF to SMILES conversion failed：Output content is not a valid SMILES record."
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    return output_file


def _write_normalization_log(output_file: Path, command: list[str], result: subprocess.CompletedProcess) -> None:
    """Save input-normalization subprocess logs."""
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if not stdout and not stderr:
        return
    log_file = output_file.with_name(f"{output_file.stem}.normalization.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(
        "\n".join(
            [
                f"command: {' '.join(command)}",
                f"returncode: {result.returncode}",
                "stdout:",
                stdout,
                "stderr:",
                stderr,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _has_valid_smiles_record(output_text: str) -> bool:
    """Validate that output contains at least one smiles-tab-name record."""
    saw_valid = False
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            return False
        smiles = parts[0].strip()
        name = parts[1].strip()
        if not name or not smiles:
            return False
        saw_valid = True
    return saw_valid
