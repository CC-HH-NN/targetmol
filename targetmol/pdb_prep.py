"""处理 PDB ID 预下载和参考配体判断，供输入准备阶段复用。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
LIGAND_DOWNLOAD_URLS = (
    "https://files.rcsb.org/ligands/download/{ligand_code}_ideal.sdf",
    "https://files.rcsb.org/ligands/download/{ligand_code}_model.sdf",
)
REJECTED_LIGANDS = {
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


@dataclass
class PreparedPdbInputs:
    """PDB ID 预处理后的结构化结果。"""

    pdb_id: str
    pdb_file: Path
    target_name: str | None
    reference_ligand: Path | None
    ligand_name: str | None
    ligand_download_error: str | None = None

    @property
    def has_reference_ligand(self) -> bool:
        """是否已拿到可用参考配体。"""
        return self.reference_ligand is not None


@dataclass(frozen=True)
class LigandRecord:
    """PDB 中一个异源小分子残基的位置记录。"""

    name: str
    chain: str
    residue_number: str
    atom_count: int


def prepare_pdb_inputs_for_run(pdb_id: str, run_dir: Path) -> PreparedPdbInputs:
    """下载 PDB 并尽量补齐靶点名与参考配体。"""
    normalized_pdb_id = pdb_id.strip().upper()
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    pdb_file = inputs_dir / f"{normalized_pdb_id}.pdb"
    try:
        pdb_text = _download_text(
            PDB_DOWNLOAD_URL.format(pdb_id=normalized_pdb_id),
            pdb_file,
        )
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"PDB ID {normalized_pdb_id} 下载失败：{exc}") from exc
    target_name = _parse_target_name_from_pdb(pdb_text)
    ligand_record = _choose_ligand_record_from_pdb(pdb_text)
    ligand_name = ligand_record.name if ligand_record else None
    reference_ligand = None
    ligand_download_error = None
    if ligand_record:
        reference_ligand = _write_cocrystal_ligand_sdf(
            pdb_text=pdb_text,
            ligand_record=ligand_record,
            pdb_id=normalized_pdb_id,
            inputs_dir=inputs_dir,
        )
    if ligand_name and reference_ligand is None:
        try:
            reference_ligand = _download_ligand_sdf(
                ligand_name=ligand_name,
                pdb_id=normalized_pdb_id,
                inputs_dir=inputs_dir,
            )
        except (HTTPError, URLError) as exc:
            ligand_download_error = str(exc)

    return PreparedPdbInputs(
        pdb_id=normalized_pdb_id,
        pdb_file=pdb_file,
        target_name=target_name,
        reference_ligand=reference_ligand,
        ligand_name=ligand_name,
        ligand_download_error=ligand_download_error,
    )


def _download_text(url: str, output_path: Path) -> str:
    """下载文本文件并写入运行目录。"""
    text = _download_bytes(url).decode("utf-8")
    output_path.write_text(text, encoding="utf-8")
    return text


def _download_ligand_sdf(ligand_name: str, pdb_id: str, inputs_dir: Path) -> Path | None:
    """尝试下载 RCSB 配体 SDF，先 ideal 再 model。"""
    output_path = inputs_dir / f"{pdb_id}_{ligand_name}.sdf"
    for template in LIGAND_DOWNLOAD_URLS:
        try:
            content = _download_bytes(template.format(ligand_code=ligand_name))
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        except URLError:
            raise
        output_path.write_bytes(content)
        return output_path
    return None


def _download_bytes(url: str) -> bytes:
    """下载远程内容。"""
    with urlopen(url) as response:
        return response.read()


def _parse_target_name_from_pdb(pdb_text: str) -> str | None:
    """从 COMPND 的 MOLECULE 段尽量提取靶点名。"""
    for line in pdb_text.splitlines():
        if not line.startswith("COMPND"):
            continue
        payload = line[10:].strip()
        upper_payload = payload.upper()
        if "MOLECULE:" not in upper_payload:
            continue
        molecule_value = payload[upper_payload.index("MOLECULE:") + len("MOLECULE:") :].strip()
        molecule_value = molecule_value.rstrip(";").strip()
        if molecule_value:
            return molecule_value
    return None


def _choose_ligand_name_from_pdb(pdb_text: str) -> str | None:
    """从异源残基里选一个最像小分子配体的候选。"""
    ligand_record = _choose_ligand_record_from_pdb(pdb_text)
    return ligand_record.name if ligand_record else None


def _choose_ligand_record_from_pdb(pdb_text: str) -> LigandRecord | None:
    """从异源残基里选一个最像小分子配体的位置记录。"""
    residue_atom_counts: dict[str, int] = {}
    residue_records: dict[str, LigandRecord] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("HETATM"):
            continue
        residue_name = line[17:20].strip().upper()
        if not _is_plausible_ligand_name(residue_name):
            continue
        chain = line[21].strip()
        residue_number = line[22:26].strip()
        residue_key = f"{residue_name}:{chain}:{residue_number}"
        residue_atom_counts[residue_key] = residue_atom_counts.get(residue_key, 0) + 1
        residue_records[residue_key] = LigandRecord(
            name=residue_name,
            chain=chain,
            residue_number=residue_number,
            atom_count=residue_atom_counts[residue_key],
        )

    best_record = None
    best_atoms = 0
    for residue_key, atom_count in residue_atom_counts.items():
        if atom_count < 6:
            continue
        if atom_count > best_atoms:
            record = residue_records[residue_key]
            best_record = LigandRecord(
                name=record.name,
                chain=record.chain,
                residue_number=record.residue_number,
                atom_count=atom_count,
            )
            best_atoms = atom_count
    return best_record


def _write_cocrystal_ligand_sdf(
    pdb_text: str,
    ligand_record: LigandRecord,
    pdb_id: str,
    inputs_dir: Path,
) -> Path | None:
    """从 PDB 共晶坐标提取参考配体并写成 SDF。"""
    ligand_lines, ligand_serials = _extract_ligand_pdb_lines(pdb_text, ligand_record)
    if not ligand_lines or not ligand_serials:
        return None
    conect_lines = _extract_ligand_conect_lines(pdb_text, ligand_serials)
    if not conect_lines:
        return None

    try:
        from rdkit import Chem
    except ImportError:
        return None

    pdb_block = "\n".join([*ligand_lines, *conect_lines, "END", ""])
    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    if mol is None:
        return None
    Chem.SanitizeMol(mol, catchErrors=True)
    mol.SetProp("_Name", f"{pdb_id}_{ligand_record.name}_{ligand_record.chain}{ligand_record.residue_number}")

    output_path = inputs_dir / f"{pdb_id}_{ligand_record.name}.sdf"
    writer = Chem.SDWriter(str(output_path))
    try:
        writer.write(mol)
    finally:
        writer.close()
    return output_path


def _extract_ligand_pdb_lines(pdb_text: str, ligand_record: LigandRecord) -> tuple[list[str], set[str]]:
    """取出指定共晶配体残基的 HETATM 行和原子序号。"""
    ligand_lines = []
    ligand_serials = set()
    for line in pdb_text.splitlines():
        if not line.startswith("HETATM"):
            continue
        if line[17:20].strip().upper() != ligand_record.name:
            continue
        if line[21].strip() != ligand_record.chain:
            continue
        if line[22:26].strip() != ligand_record.residue_number:
            continue
        ligand_lines.append(line)
        ligand_serials.add(line[6:11].strip())
    return ligand_lines, ligand_serials


def _extract_ligand_conect_lines(pdb_text: str, ligand_serials: set[str]) -> list[str]:
    """取出指定配体内部 CONECT 记录。"""
    conect_lines = []
    for line in pdb_text.splitlines():
        if not line.startswith("CONECT"):
            continue
        serials = line.split()[1:]
        if not serials:
            continue
        if serials[0] in ligand_serials:
            conect_lines.append(line)
    return conect_lines


def _is_plausible_ligand_name(residue_name: str) -> bool:
    """过滤明显不是小分子参考配体的残基名。"""
    if not residue_name:
        return False
    if residue_name in REJECTED_LIGANDS:
        return False
    if len(residue_name) > 3:
        return False
    return residue_name.isalnum()
