"""解析与标准化 TargetMol 的输入参数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class InputSpec:
    """一次运行所需的标准化输入。"""

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
        """是否具备结构输入。"""
        return bool(self.pdb_id or self.pdb_file)

    @property
    def has_reference_ligand(self) -> bool:
        """是否具备参考配体。"""
        return self.reference_ligand is not None

    @property
    def has_seed_smiles(self) -> bool:
        """是否具备 seed SMILES 文件。"""
        return self.seed_smiles_file is not None

    @property
    def has_target_context(self) -> bool:
        """是否具备 ligand 生成可用的文本上下文。"""
        return self.target_name is not None


def build_input_spec(args) -> InputSpec:
    """从 argparse 结果构建统一输入对象。"""
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
    """把可选路径参数标准化为绝对路径。"""
    if not path_value:
        return None
    return Path(path_value).expanduser().resolve()


def _resolve_candidate_file_arg(args) -> Path | None:
    """解析候选文件参数，并阻止冲突输入被静默吞掉。"""
    candidate_file = _resolve_path(getattr(args, "candidate_file", None))
    candidate_smiles_file = _resolve_path(getattr(args, "candidate_smiles_file", None))
    if candidate_file and candidate_smiles_file and candidate_file != candidate_smiles_file:
        raise ValueError("`--candidate-file` 和 `--candidate-smiles-file` 不能同时指向不同文件。")
    return candidate_file or candidate_smiles_file
