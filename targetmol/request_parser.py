"""Parse natural-language requests into TargetMol input fields."""

from __future__ import annotations

import re
from pathlib import Path

from targetmol.inputs import InputSpec


PDB_ID_PATTERN = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")


def extract_pdb_id(request_text: str) -> str | None:
    """Extract a PDB ID from a natural-language request."""
    match = PDB_ID_PATTERN.search(request_text)
    if match is None:
        return None
    return match.group(1).upper()


def build_input_spec_from_request(
    request_text: str,
    *,
    pdb_id: str | None = None,
    pdb_file: Path | None = None,
    reference_ligand: Path | None = None,
    seed_smiles_file: Path | None = None,
    candidate_smiles_file: Path | None = None,
    target_name: str | None = None,
    disease: str | None = None,
    run_name: str,
) -> InputSpec:
    """Build an input object from request text and explicit file arguments."""
    return InputSpec(
        pdb_id=pdb_id or extract_pdb_id(request_text),
        pdb_file=pdb_file,
        reference_ligand=reference_ligand,
        seed_smiles_file=seed_smiles_file,
        candidate_smiles_file=candidate_smiles_file,
        target_name=target_name,
        disease=disease,
        run_name=run_name,
        request_text=request_text,
    )
