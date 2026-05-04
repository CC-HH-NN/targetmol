# Case: Natural-Language Ligand Agent

Use this case when the user only has a target or disease-level request and no reliable structure input.

```bash
.venv/bin/python -m targetmol.cli \
  --config targetmol.yaml \
  --request "Design EGFR inhibitors for lung cancer" \
  --run-name example_ligand_agent \
  --dry-run
```

Expected route:

```text
ligand_based_targetmol
```

A full run performs request understanding, search-grounded target context extraction, embedding-ranked evidence ordering, candidate expansion, iterative molecular refinement, and agent report writing.
