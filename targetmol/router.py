"""根据输入条件选择 TargetMol 生成路线。"""


def choose_route(
    *,
    has_pdb: bool,
    has_reference_ligand: bool,
    has_seed_smiles: bool,
    has_target_context: bool,
) -> str:
    """按第一版 MVP 规则选择生成路线。"""
    if has_pdb and has_reference_ligand:
        return "sbdd_drugflow"
    if has_seed_smiles or has_target_context:
        return "ligand_based_targetmol"
    raise ValueError("无法根据当前输入选择可用路线。")
