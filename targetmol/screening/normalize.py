"""Screening input loading and normalization."""

from __future__ import annotations

from pathlib import Path

from targetmol.screening.types import ScreeningCandidate


def load_normalized_smiles_file(path: Path) -> list[ScreeningCandidate]:
    """Read a normalized .smi file into candidate objects."""
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
            raise ValueError(f"{path} line {index} is not a valid normalized SMILES record.")
        smiles = parts[0].strip()
        name = parts[1].strip()
        if not smiles or not name:
            raise ValueError(f"{path} line {index} is not a valid normalized SMILES record.")
        records.append(ScreeningCandidate(name=name, smiles=smiles))
    return records
