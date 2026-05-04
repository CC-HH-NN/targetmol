"""TargetMol 配置数据结构，供各模块统一复用。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    """项目级配置。"""

    name: str
    runs_dir: Path


@dataclass
class PathsConfig:
    """外部仓库路径配置。"""

    drugflow_root: Path


@dataclass
class EnvsConfig:
    """运行环境名称配置。"""

    drugflow_conda_env: str
    screening_conda_env: str


@dataclass
class ModelsConfig:
    """聊天与嵌入模型配置。"""

    chat_provider: str
    chat_base_url: str
    chat_model: str
    chat_api_key: str
    embedding_provider: str
    embedding_base_url: str
    embedding_model: str
    embedding_api_key: str
    chat_thinking: str = "disabled"


@dataclass
class DrugFlowConfig:
    """DrugFlow 运行配置。"""

    checkpoint: Path
    device: str
    n_samples: int
    batch_size: int
    n_steps: int
    pocket_distance_cutoff: float


@dataclass
class ScreeningConfig:
    """TargetMol 自研筛选配置。"""

    top_k: int
    exhaustiveness: int
    n_poses: int
    n_threads: int


@dataclass
class SearchConfig:
    """联网检索配置。"""

    serper_api_key: str


@dataclass
class LigandGenerationConfig:
    """TargetMol ligand 生成配置。"""

    iterations: int
    num_smiles: int


@dataclass
class ToolsConfig:
    """外部工具路径配置。"""

    vina: str
    gnina: str
    reduce: str
    wget: str


@dataclass
class TargetMolConfig:
    """TargetMol 根配置对象。"""

    project: ProjectConfig
    paths: PathsConfig
    envs: EnvsConfig
    models: ModelsConfig
    drugflow: DrugFlowConfig
    screening: ScreeningConfig
    search: SearchConfig
    ligand_generation: LigandGenerationConfig
    tools: ToolsConfig
