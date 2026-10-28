"""
Unified Experiment Runner Module (Blueprint Section 15).
Provides programmatic interface to run baselines (B0-B5), proposed method (B6),
and ablation sweeps under frozen configs and deterministic seeds.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.common.utils import load_config, set_seed

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Master runner executing evaluations across frozen configs and splits.
    """

    def __init__(self, config_path: str = "configs/base.yaml"):
        self.config = load_config(config_path)
        self.seed = self.config.get("seed", 42)
        set_seed(self.seed)

        self.raw_dir = self.config.get("paths", {}).get("results_raw_dir", "results/raw")
        self.tables_dir = self.config.get("paths", {}).get("results_tables_dir", "results/tables")
        self.splits_dir = self.config.get("paths", {}).get("splits_dir", "data/splits")

    def load_split(self, split_name: str = "test") -> List[str]:
        """Load frozen question IDs for train, val, or test split."""
        split_file = os.path.join(self.splits_dir, f"{split_name}_ids.json")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r") as f:
            return json.load(f)

    def load_baseline_outputs(self, baseline_id: str) -> Dict[str, Dict[str, Any]]:
        """Load per-query outputs for a baseline."""
        filename = f"{baseline_id.lower()}_outputs.json"
        path = os.path.join(self.raw_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Baseline raw output file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        return {item["question_id"]: item for item in data}

    def get_summary_table(self, split_name: Optional[str] = "test") -> pd.DataFrame:
        """Retrieve aggregated performance metrics table."""
        if split_name == "test":
            csv_path = os.path.join(self.tables_dir, "test_split_summary.csv")
        else:
            csv_path = os.path.join(self.tables_dir, "baselines_summary.csv")

        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"Summary table not found: {csv_path}")
