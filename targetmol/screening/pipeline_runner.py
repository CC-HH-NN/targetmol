"""Internal screening subcommand entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from targetmol.screening.pipeline import run_screening_pipeline


def main(argv: list[str] | None = None) -> int:
    """Run the screening runner."""
    parser = argparse.ArgumentParser(description="TargetMol screening runner")
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--receptor-pdb", required=True)
    parser.add_argument("--reference-ligand", required=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gnina", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--exhaustiveness", type=int, required=True)
    parser.add_argument("--n-poses", type=int, required=True)
    parser.add_argument("--n-threads", type=int, required=True)
    parser.add_argument("--box-size", nargs=3, type=float, default=(22.0, 22.0, 22.0))
    args = parser.parse_args(argv)

    run_screening_pipeline(
        normalized_smiles_file=Path(args.smiles_file),
        receptor_pdb=Path(args.receptor_pdb),
        reference_ligand=Path(args.reference_ligand) if args.reference_ligand else None,
        output_dir=Path(args.output_dir),
        gnina_bin=args.gnina,
        top_k=args.top_k,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        n_threads=args.n_threads,
        box_size=(args.box_size[0], args.box_size[1], args.box_size[2]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
