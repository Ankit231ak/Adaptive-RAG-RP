"""
Shared utilities for Adaptive-RAG.
"""
import os
import random
import yaml
import numpy as np


def load_config(config_path: str = "configs/base.yaml") -> dict:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
