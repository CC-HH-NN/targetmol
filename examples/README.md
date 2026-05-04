# Examples

This directory contains small public input examples and case recipes for dry-run checks and quick demos.

Layout:

- `inputs/smiles_examples.txt`: a tiny SMILES list for `screen_only` planning.
- `inputs/requests.txt`: natural-language task prompts that exercise TargetMol routing.
- `cases/screen_only.md`: uploaded-candidate screening case.
- `cases/sbdd_pdb_id.md`: PDB ID structure-driven generation case.
- `cases/ligand_based.md`: natural-language ligand-agent case.
- `results/`: sanitized excerpts from real server runs, including summaries, compact candidate tables, and public metadata.

The result files are public examples only. They are not experimental results and should not be interpreted as validated drug candidates.

Example dry-run:

```bash
.venv/bin/python -m targetmol.cli \
  --config targetmol.yaml \
  --pdb-id 6JX0 \
  --candidate-smiles-file examples/inputs/smiles_examples.txt \
  --run-name example_screen \
  --dry-run
```
