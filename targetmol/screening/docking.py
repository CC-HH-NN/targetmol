"""clean-room docking 命令构建。"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from targetmol.screening.types import ScreeningCandidate
from targetmol.pdb_prep import LigandRecord


IGNORED_PDB_LIGANDS = {
    "HOH",
    "WAT",
    "DOD",
    "SO4",
    "PO4",
    "PEG",
    "GOL",
    "EDO",
    "ACT",
    "FMT",
    "CL",
    "BR",
    "IOD",
    "NA",
    "K",
    "CA",
    "MG",
    "ZN",
    "MN",
    "CO",
    "NI",
    "CU",
    "FE",
    "CD",
}


def build_gnina_command(
    *,
    gnina_bin: str,
    receptor_pdb: Path,
    ligand_sdf: Path,
    out_sdf: Path,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    exhaustiveness: int,
    n_poses: int,
    n_threads: int,
) -> list[str]:
    """构建 gnina 对接命令。"""
    return [
        gnina_bin,
        "--receptor",
        str(receptor_pdb),
        "--ligand",
        str(ligand_sdf),
        "--out",
        str(out_sdf),
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(box_size[0]),
        "--size_y",
        str(box_size[1]),
        "--size_z",
        str(box_size[2]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(n_poses),
        "--cpu",
        str(n_threads),
    ]


def _safe_filename(name: str) -> str:
    """把候选名称转换成稳定文件名。"""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "candidate"


def _load_rdkit():
    """延迟加载 RDKit，避免无关路径在导入时直接失败。"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError("RDKit 不可用，无法为 clean-room screening 准备 3D 配体。") from exc
    return Chem, AllChem


def _compute_center_from_reference_ligand(reference_ligand: Path) -> tuple[float, float, float]:
    """根据参考配体 3D 坐标计算 docking 中心。"""
    Chem, _ = _load_rdkit()
    supplier = Chem.SDMolSupplier(str(reference_ligand), removeHs=False)
    mol = supplier[0] if supplier and len(supplier) > 0 else None
    if mol is None or mol.GetNumConformers() == 0:
        raise RuntimeError(f"无法从参考配体读取 3D 坐标: {reference_ligand}")

    conf = mol.GetConformer()
    coordinates = []
    for atom_index in range(mol.GetNumAtoms()):
        position = conf.GetAtomPosition(atom_index)
        coordinates.append((float(position.x), float(position.y), float(position.z)))
    return _average_coordinates(coordinates)


def _choose_ligand_record_from_pdb_text(pdb_text: str) -> LigandRecord | None:
    """从 PDB 文本中选一个最像共晶配体的残基记录。"""
    residue_atoms: dict[str, list[tuple[float, float, float]]] = {}
    residue_records: dict[str, LigandRecord] = {}
    for line in pdb_text.splitlines():
        residue = _parse_pdb_ligand_atom(line)
        if residue is None:
            continue
        residue_name, chain, residue_number, coords = residue
        residue_key = _pdb_residue_key(residue_name, chain, residue_number)
        residue_atoms.setdefault(residue_key, []).append(coords)
        residue_records[residue_key] = LigandRecord(
            name=residue_name,
            chain=chain,
            residue_number=residue_number,
            atom_count=len(residue_atoms[residue_key]),
        )

    best_record: LigandRecord | None = None
    for residue_key, atoms in residue_atoms.items():
        if len(atoms) < 6:
            continue
        record = residue_records[residue_key]
        record = LigandRecord(
            name=record.name,
            chain=record.chain,
            residue_number=record.residue_number,
            atom_count=len(atoms),
        )
        if best_record is None or record.atom_count > best_record.atom_count:
            best_record = record
    return best_record


def _compute_center_from_receptor_pdb(receptor_pdb: Path) -> tuple[float, float, float]:
    """从受体 PDB 中共晶配体坐标推断 docking 中心。"""
    pdb_text = receptor_pdb.read_text(encoding="utf-8", errors="ignore")
    ligand_record = _choose_ligand_record_from_pdb_text(pdb_text)
    if ligand_record is None:
        raise RuntimeError(f"无法从受体 PDB 自动识别可用共晶配体中心: {receptor_pdb}")

    coordinates: list[tuple[float, float, float]] = []
    for line in pdb_text.splitlines():
        residue = _parse_pdb_ligand_atom(line)
        if residue is None:
            continue
        residue_name, chain, residue_number, coords = residue
        if _pdb_residue_key(residue_name, chain, residue_number) == _ligand_record_key(ligand_record):
            coordinates.append(coords)
    if not coordinates:
        raise RuntimeError(f"无法从受体 PDB 读取共晶配体坐标: {receptor_pdb}")
    return _average_coordinates(coordinates)


def _parse_pdb_ligand_atom(line: str) -> tuple[str, str, str, tuple[float, float, float]] | None:
    """解析一个可用于口袋中心推断的 HETATM 记录。"""
    if not line.startswith("HETATM"):
        return None

    residue_name = line[17:20].strip().upper()
    if not _is_candidate_residue_name(residue_name):
        return None

    try:
        coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except ValueError:
        return None
    return residue_name, line[21].strip(), line[22:26].strip(), coords


def _is_candidate_residue_name(residue_name: str) -> bool:
    """判断 PDB 残基名是否可能代表共晶配体。"""
    return (
        bool(residue_name)
        and residue_name not in IGNORED_PDB_LIGANDS
        and len(residue_name) <= 3
        and residue_name.isalnum()
    )


def _pdb_residue_key(residue_name: str, chain: str, residue_number: str) -> str:
    """生成 PDB 残基的稳定匹配键。"""
    return f"{residue_name}:{chain}:{residue_number}"


def _ligand_record_key(record: LigandRecord) -> str:
    """生成 LigandRecord 的稳定匹配键。"""
    return _pdb_residue_key(record.name, record.chain, record.residue_number)


def _average_coordinates(coordinates: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """计算 3D 坐标中心。"""
    count = len(coordinates)
    return (
        sum(x for x, _, _ in coordinates) / count,
        sum(y for _, y, _ in coordinates) / count,
        sum(z for _, _, z in coordinates) / count,
    )


def _resolve_docking_center(
    receptor_pdb: Path,
    reference_ligand: Path | None,
) -> tuple[float, float, float]:
    """优先用参考配体中心，缺失时回退到受体中的共晶配体中心。"""
    if reference_ligand is not None:
        return _compute_center_from_reference_ligand(reference_ligand)
    return _compute_center_from_receptor_pdb(receptor_pdb)


def _prepare_candidate_ligand(candidate: ScreeningCandidate, ligand_dir: Path) -> Path:
    """把 SMILES 转成带 3D 坐标的 SDF，供 gnina 对接。"""
    Chem, AllChem = _load_rdkit()
    mol = Chem.MolFromSmiles(candidate.smiles)
    if mol is None:
        raise RuntimeError(f"RDKit 无法解析候选 SMILES: {candidate.smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"RDKit 无法为候选生成 3D 构象: {candidate.name}")

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        # 构象已经存在时，优化失败不阻止后续 docking。
        pass

    ligand_dir.mkdir(parents=True, exist_ok=True)
    ligand_path = ligand_dir / f"{_safe_filename(candidate.name)}.sdf"
    writer = Chem.SDWriter(str(ligand_path))
    if writer is None:
        raise RuntimeError(f"无法创建候选 SDF 文件: {ligand_path}")
    writer.write(mol)
    writer.close()
    return ligand_path


def _write_command_record(commands_path: Path, payload: dict[str, object]) -> None:
    """把单次 docking 命令记录到 jsonl，便于追溯。"""
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    with commands_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _extract_sdf_property(sdf_path: Path, property_name: str) -> str | None:
    """从 SDF 文本中提取首个属性值。"""
    if not sdf_path.exists():
        return None
    lines = sdf_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    marker_pattern = re.compile(rf"^>\s*<{re.escape(property_name)}>\s*$")
    for index, line in enumerate(lines):
        if marker_pattern.match(line.strip()) and index + 1 < len(lines):
            value = lines[index + 1].strip()
            if value:
                return value
    return None


def _extract_score_from_stdout(stdout: str) -> float | None:
    """优先从 gnina 输出中提取可比较的 docking 分数。"""
    for label in ("minimizedAffinity", "CNNaffinity", "CNNscore"):
        for line in stdout.splitlines():
            if line.startswith(label):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1].strip())
                    except ValueError:
                        continue
    return None


def _read_docking_score(pose_sdf: Path, stdout: str) -> float | None:
    """优先从 pose SDF 读取分数，缺失时再回退到 stdout。"""
    raw_score = _extract_sdf_property(pose_sdf, "minimizedAffinity")
    if raw_score is not None:
        try:
            return float(raw_score)
        except ValueError:
            pass
    return _extract_score_from_stdout(stdout)


def _failed_row(candidate: ScreeningCandidate, error: str) -> dict[str, object]:
    """构造稳定的失败结果行。"""
    return {
        "name": candidate.name,
        "smiles": candidate.smiles,
        "docking_score": None,
        "pose_sdf": None,
        "error": error,
    }


def _build_gnina_runtime_env() -> dict[str, str]:
    """为 gnina 准备运行时环境，补上当前 conda 环境的动态库路径。"""
    runtime_env = dict(os.environ)
    conda_prefix = runtime_env.get("CONDA_PREFIX", "").strip()
    if not conda_prefix:
        return runtime_env

    conda_lib = str(Path(conda_prefix) / "lib")
    current = runtime_env.get("LD_LIBRARY_PATH", "").strip()
    if not current:
        runtime_env["LD_LIBRARY_PATH"] = conda_lib
        return runtime_env

    entries = [item for item in current.split(":") if item]
    if conda_lib not in entries:
        runtime_env["LD_LIBRARY_PATH"] = ":".join([conda_lib, *entries])
    return runtime_env


def run_docking_batch(
    *,
    candidates: list[ScreeningCandidate],
    receptor_pdb: Path,
    reference_ligand: Path | None,
    output_dir: Path,
    gnina_bin: str,
    exhaustiveness: int,
    n_poses: int,
    n_threads: int,
    box_size: tuple[float, float, float],
) -> list[dict[str, object]]:
    """逐个执行 gnina docking，并保存中间文件与日志。"""
    docking_dir = output_dir / "docking"
    ligand_dir = docking_dir / "ligands"
    poses_dir = docking_dir / "poses"
    logs_dir = output_dir / "logs"
    commands_path = docking_dir / "commands.jsonl"
    results_path = docking_dir / "docking_results.json"

    ligand_dir.mkdir(parents=True, exist_ok=True)
    poses_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        center = _resolve_docking_center(receptor_pdb, reference_ligand)
    except Exception as exc:
        results = [_failed_row(candidate, str(exc)) for candidate in candidates]
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return results

    results: list[dict[str, object]] = []
    for candidate in candidates:
        safe_name = _safe_filename(candidate.name)
        try:
            ligand_sdf = _prepare_candidate_ligand(candidate, ligand_dir)
        except Exception as exc:
            results.append(_failed_row(candidate, str(exc)))
            continue

        pose_path = poses_dir / f"{safe_name}.sdf"
        command = build_gnina_command(
            gnina_bin=gnina_bin,
            receptor_pdb=receptor_pdb,
            ligand_sdf=ligand_sdf,
            out_sdf=pose_path,
            center=center,
            box_size=box_size,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            n_threads=n_threads,
        )
        _write_command_record(
            commands_path,
            {
                "name": candidate.name,
                "command": command,
                "receptor_pdb": str(receptor_pdb),
                "reference_ligand": str(reference_ligand) if reference_ligand is not None else None,
                "input_ligand": str(ligand_sdf),
                "pose_sdf": str(pose_path),
            },
        )

        result = subprocess.run(
            command,
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            env=_build_gnina_runtime_env(),
        )
        (logs_dir / f"{safe_name}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (logs_dir / f"{safe_name}.stderr.log").write_text(result.stderr, encoding="utf-8")

        if result.returncode != 0:
            results.append(_failed_row(candidate, f"gnina 失败(returncode={result.returncode}): {result.stderr.strip()}"))
            continue

        score = _read_docking_score(pose_path, result.stdout)
        if score is None:
            results.append(_failed_row(candidate, "gnina 已完成，但未能解析 docking score。"))
            continue

        results.append(
            {
                "name": candidate.name,
                "smiles": candidate.smiles,
                "docking_score": score,
                "pose_sdf": str(pose_path.relative_to(output_dir)),
                "error": None,
            }
        )

    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results
