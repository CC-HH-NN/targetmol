"""TargetMol configuration data structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    """Project-level configuration."""

    name: str
    runs_dir: Path


@dataclass
class PathsConfig:
    """External repository path configuration."""

    drugflow_root: Path


@dataclass
class EnvsConfig:
    """Runtime environment name configuration."""

    drugflow_conda_env: str
    screening_conda_env: str


@dataclass
class ModelsConfig:
    """Chat and embedding model configuration."""

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
    """DrugFlow runtime configuration."""

    checkpoint: Path
    device: str
    n_samples: int
    batch_size: int
    n_steps: int
    pocket_distance_cutoff: float


@dataclass
class ScreeningConfig:
    """TargetMol screening configuration."""

    top_k: int
    exhaustiveness: int
    n_poses: int
    n_threads: int


@dataclass
class SearchConfig:
    """Web search configuration."""

    serper_api_key: str


@dataclass
class LigandGenerationConfig:
    """TargetMol ligand generation configuration."""

    iterations: int
    num_smiles: int


@dataclass
class ToolsConfig:
    """External tool path configuration."""

    vina: str
    gnina: str
    reduce: str
    wget: str


@dataclass
class TargetMolConfig:
    """Root TargetMol configuration object."""

    project: ProjectConfig
    paths: PathsConfig
    envs: EnvsConfig
    models: ModelsConfig
    drugflow: DrugFlowConfig
    screening: ScreeningConfig
    search: SearchConfig
    ligand_generation: LigandGenerationConfig
    tools: ToolsConfig
