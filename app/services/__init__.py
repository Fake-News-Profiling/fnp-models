from typing import Dict
from dataclasses import dataclass


@dataclass
class ModelConfig:
    load_path: str


@dataclass
class ServiceConfig:
    models: Dict[str, ModelConfig]