# Case: PDB ID Structure-Driven Generation

Use this case when a target structure can provide a reference ligand.

```bash
.venv/bin/python -m targetmol.cli \
  --config targetmol.yaml \
  --pdb-id 6JX0 \
  --request "Design EGFR inhibitors" \
  --run-name example_6jx0_sbdd \
  --dry-run
```

Expected route when a usable co-crystal ligand is found:

```text
sbdd_drugflow
```

A full run performs PDB preparation, reference ligand extraction, DrugFlow generation, SDF-to-SMILES conversion, TargetMol screening, and final report writing.
