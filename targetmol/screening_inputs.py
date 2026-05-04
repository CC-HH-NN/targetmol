"""筛选输入标准化模块，供工作流在执行前统一候选分子文件。"""

from __future__ import annotations

import subprocess
from pathlib import Path

SMILES_SUFFIXES = {".smi", ".smiles", ".txt"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_screening_input_file(candidate_file: Path, run_dir: Path, conda_env: str) -> Path:
    """把候选 SMILES 或 SDF 文件统一整理成 normalized 下的 .smi。"""
    source = candidate_file.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"候选分子文件不存在: {source}")

    normalized_dir = run_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    output_file = normalized_dir / f"{source.stem}.smi"
    suffix = source.suffix.lower()

    if suffix in SMILES_SUFFIXES:
        return _normalize_smiles_like_file(source, output_file, conda_env)
    if suffix == ".sdf":
        return _convert_sdf_to_smiles(source, output_file, conda_env)

    raise ValueError("不支持的候选文件后缀，仅支持 .smi/.smiles/.txt/.sdf。")


def _normalize_smiles_like_file(source: Path, output_file: Path, conda_env: str) -> Path:
    """把文本候选文件统一整理成 smiles-tab-name 记录。"""
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
            "SMILES 文本标准化失败: "
            f"{source}; command={' '.join(command)}; "
            f"stdout={result.stdout.strip()}; stderr={result.stderr.strip()}"
        )
    if not output_file.exists():
        raise RuntimeError(
            "SMILES 文本标准化失败：未生成输出文件。"
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    output_text = output_file.read_text(encoding="utf-8")
    if not output_text.strip():
        raise RuntimeError(
            "SMILES 文本标准化失败：没有生成可用分子。"
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    if not _has_valid_smiles_record(output_text):
        raise RuntimeError(
            "SMILES 文本标准化失败：输出内容不是有效的 SMILES 记录。"
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    return output_file


def _convert_sdf_to_smiles(source: Path, output_file: Path, conda_env: str) -> Path:
    """把 SDF 候选文件转换成标准 .smi。"""
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
            "SDF 转 SMILES 失败: "
            f"{source}; command={' '.join(command)}; "
            f"stdout={result.stdout.strip()}; stderr={result.stderr.strip()}"
        )
    if not output_file.exists():
        raise RuntimeError(
            "SDF 转 SMILES 失败：未生成输出文件。"
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    output_text = output_file.read_text(encoding="utf-8")
    if not output_text.strip():
        raise RuntimeError(
            "SDF 转 SMILES 失败：没有生成可用分子。"
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    if not _has_valid_smiles_record(output_text):
        raise RuntimeError(
            "SDF 转 SMILES 失败：输出内容不是有效的 SMILES 记录。"
            f" input={source}; output={output_file}; command={' '.join(command)}"
        )
    return output_file


def _write_normalization_log(output_file: Path, command: list[str], result: subprocess.CompletedProcess) -> None:
    """保存输入标准化子进程日志，方便追踪跳过记录和 RDKit 警告。"""
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
    """检查输出是否全部为有效的 smiles-tab-name 记录，且至少有一条。"""
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
