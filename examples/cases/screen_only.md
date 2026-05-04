# Case: Screen Uploaded Candidates

Use this case when you already have candidate molecules and only want screening.

```bash
.venv/bin/python -m targetmol.cli \
  --config targetmol.yaml \
  --request "Screen uploaded EGFR inhibitors for lung cancer" \
  --pdb-id 6JX0 \
  --candidate-smiles-file examples/inputs/smiles_examples.txt \
  --run-name example_screen_only \
  --dry-run
```

Expected route:

```text
screen_only
```

A full run performs candidate normalization, docking, property scoring, ranking, and report writing.
