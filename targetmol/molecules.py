"""Handle candidate molecule records and SMILES files."""

from __future__ import annotations

from pathlib import Path


def deduplicate_smiles_records(records: list[dict]) -> list[dict]:
    """Deduplicate records by SMILES while preserving first occurrence."""
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
    """Write candidates as name-tab-smiles records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"{record['name']}\t{record['smiles']}\n")
    return output_path
