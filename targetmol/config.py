"""读取 TargetMol 统一 YAML 配置并转换为强类型对象。"""

from pathlib import Path

import yaml

from targetmol.models import (
    DrugFlowConfig,
    EnvsConfig,
    LigandGenerationConfig,
    ModelsConfig,
    PathsConfig,
    ProjectConfig,
    SearchConfig,
    ScreeningConfig,
    TargetMolConfig,
    ToolsConfig,
)


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    """把配置中的路径解析为绝对路径，支持相对路径写法。"""
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_screening_config(data: dict) -> ScreeningConfig:
    """读取当前正式的 screening 配置段。"""
    raw = data["screening"]
    return ScreeningConfig(
        top_k=raw["top_k"],
        exhaustiveness=raw["exhaustiveness"],
        n_poses=raw["n_poses"],
        n_threads=raw["n_threads"],
    )


def load_config(path: str | Path) -> TargetMolConfig:
    """从 YAML 文件加载 TargetMol 配置。"""
    config_path = Path(path).expanduser().resolve()
    base_dir = config_path.parent
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return TargetMolConfig(
        project=ProjectConfig(
            name=data["project"]["name"],
            runs_dir=_resolve_path(base_dir, data["project"]["runs_dir"]),
        ),
        paths=PathsConfig(
            drugflow_root=_resolve_path(base_dir, data["paths"]["drugflow_root"]),
        ),
        envs=EnvsConfig(**data["envs"]),
        models=ModelsConfig(**data["models"]),
        drugflow=DrugFlowConfig(
            checkpoint=_resolve_path(base_dir, data["drugflow"]["checkpoint"]),
            device=data["drugflow"]["device"],
            n_samples=data["drugflow"]["n_samples"],
            batch_size=data["drugflow"]["batch_size"],
            n_steps=data["drugflow"]["n_steps"],
            pocket_distance_cutoff=data["drugflow"]["pocket_distance_cutoff"],
        ),
        screening=_load_screening_config(data),
        search=SearchConfig(
            serper_api_key=data["search"]["serper_api_key"],
        ),
        ligand_generation=LigandGenerationConfig(
            iterations=data["ligand_generation"]["iterations"],
            num_smiles=data["ligand_generation"]["num_smiles"],
        ),
        tools=ToolsConfig(**data["tools"]),
    )
