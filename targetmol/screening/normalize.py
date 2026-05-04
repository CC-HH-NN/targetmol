"""clean-room screening 输入加载与规范化。"""

from __future__ import annotations

from pathlib import Path

from targetmol.screening.types import ScreeningCandidate


def load_normalized_smiles_file(path: Path) -> list[ScreeningCandidate]:
    """读取 normalized .smi 文件并转成候选对象。"""
    records: list[ScreeningCandidate] = []
    for index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if "\t" in line:
            parts = line.split("\t", 1)
        else:
            parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{path} 第 {index} 行不是有效的 normalized SMILES 记录。")
        smiles = parts[0].strip()
        name = parts[1].strip()
        if not smiles or not name:
            raise ValueError(f"{path} 第 {index} 行不是有效的 normalized SMILES 记录。")
        records.append(ScreeningCandidate(name=name, smiles=smiles))
    return records
