"""Parse and normalize TargetMol input arguments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class InputSpec:
    """Normalized inputs for one run."""

    pdb_id: str | None
    pdb_file: Path | None
    reference_ligand: Path | None
    seed_smiles_file: Path | None
    candidate_smiles_file: Path | None
    target_name: str | None
    disease: str | None
    run_name: str
    request_text: str | None = None
    generated_candidate_source: str | None = None

    @property
    def has_pdb(self) -> bool:
        """Return whether structure input is available."""
        return bool(self.pdb_id or self.pdb_file)

    @property
    def has_reference_ligand(self) -> bool:
        """Return whether a reference ligand is available."""
        return self.reference_ligand is not None

    @property
    def has_seed_smiles(self) -> bool:
        """Return whether a seed SMILES file is available."""
        return self.seed_smiles_file is not None

    @property
    def has_target_context(self) -> bool:
        """Return whether ligand-generation text context is available."""
        return self.target_name is not None


def build_input_spec(args) -> InputSpec:
    """Build a unified input object from argparse results."""
    pdb_file = _resolve_path(getattr(args, "pdb_file", None))
    reference_ligand = _resolve_path(getattr(args, "reference_ligand", None))
    seed_smiles_file = _resolve_path(getattr(args, "seed_smiles_file", None))
    candidate_smiles_file = _resolve_candidate_file_arg(args)
    request_text = getattr(args, "request", None)

    if request_text:
        from targetmol.request_parser import build_input_spec_from_request

        return build_input_spec_from_request(
            request_text=request_text,
            pdb_id=getattr(args, "pdb_id", None),
            pdb_file=pdb_file,
            reference_ligand=reference_ligand,
            seed_smiles_file=seed_smiles_file,
            candidate_smiles_file=candidate_smiles_file,
            target_name=getattr(args, "target_name", None),
            disease=getattr(args, "disease", None),
            run_name=args.run_name,
        )

    return InputSpec(
        pdb_id=getattr(args, "pdb_id", None),
        pdb_file=pdb_file,
        reference_ligand=reference_ligand,
        seed_smiles_file=seed_smiles_file,
        candidate_smiles_file=candidate_smiles_file,
        target_name=getattr(args, "target_name", None),
        disease=getattr(args, "disease", None),
        run_name=args.run_name,
        request_text=None,
    )


def _resolve_path(path_value: str | None) -> Path | None:
    """Normalize an optional path argument to an absolute path."""
    if not path_value:
        return None
    return Path(path_value).expanduser().resolve()


def _resolve_candidate_file_arg(args) -> Path | None:
    """Resolve candidate file arguments and reject conflicting inputs."""
    candidate_file = _resolve_path(getattr(args, "candidate_file", None))
    candidate_smiles_file = _resolve_path(getattr(args, "candidate_smiles_file", None))
    if candidate_file and candidate_smiles_file and candidate_file != candidate_smiles_file:
        raise ValueError("`--candidate-file` and `--candidate-smiles-file` cannot point to different files at the same time.")
    return candidate_file or candidate_smiles_file
