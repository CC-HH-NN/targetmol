"""Select a TargetMol route from the available inputs."""


def choose_route(
    *,
    has_pdb: bool,
    has_reference_ligand: bool,
    has_seed_smiles: bool,
    has_target_context: bool,
) -> str:
    """Select a route according to the current supported input rules."""
    if has_pdb and has_reference_ligand:
        return "sbdd_drugflow"
    if has_seed_smiles or has_target_context:
        return "ligand_based_targetmol"
    raise ValueError("Unable to select an available route from the current inputs.")
