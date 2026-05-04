"""整理候选分子记录与 SMILES 文件。"""

from __future__ import annotations

from pathlib import Path


def deduplicate_smiles_records(records: list[dict]) -> list[dict]:
    """按 SMILES 去重，保留首次出现的记录。"""
    seen: set[str] = set()
    unique: list[dict] = []
    for record in records:
        smiles = record["smiles"].strip()
        if smiles in seen:
            continue
        seen.add(smiles)
        unique.append({**record, "smiles": smiles})
    return unique


def write_smiles_file(records: list[dict], output_path: Path) -> Path:
    """把候选分子写成 name-tab-smiles 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"{record['name']}\t{record['smiles']}\n")
    return output_path
