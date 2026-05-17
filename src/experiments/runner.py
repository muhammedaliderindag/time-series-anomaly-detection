import os
import json
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any

from src.utils.config_parser import ConfigParser

class MultiSeedRunner:
    """
    Wrapper to execute a training/evaluation function multiple times
    with different random seeds to ensure statistical reliability.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)
        self.seeds = self.cfg.get("random_seeds", [42, 123, 2026, 7, 999])
        self.log_dir = self.cfg.get("paths.log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def run(self, experiment_name: str, eval_func: Callable[[int], Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Runs the evaluation function for each seed and aggregates the results.
        
        Args:
            experiment_name: Name of the experiment (used for log file naming).
            eval_func: Function that takes a random seed (int) and returns a dictionary
                       of metrics (e.g., {"accuracy": 0.9, "f1": 0.85, ...}).
                       
        Returns:
            Dictionary with aggregated mean and std for each metric.
        """
        all_metrics = []
        
        print(f"--- Starting Multi-Seed Execution for '{experiment_name}' ---")
        for i, seed in enumerate(self.seeds):
            print(f"Run {i+1}/{len(self.seeds)} with seed: {seed}")
            
            # The eval_func should ideally set the global random seeds (numpy, torch, random) internally
            # or use the passed seed to initialize its own generators.
            metrics = eval_func(seed)
            metrics["seed"] = seed
            all_metrics.append(metrics)
            
        # Aggregate results
        aggregated = {}
        df = pd.DataFrame(all_metrics)
        
        # We only aggregate metric columns (ignore 'seed' and any non-numeric info)
        metric_cols = [col for col in df.columns if col != "seed" and pd.api.types.is_numeric_dtype(df[col])]
        
        for col in metric_cols:
            aggregated[f"{col}_mean"] = float(df[col].mean())
            aggregated[f"{col}_std"] = float(df[col].std())
            
        print(f"\n--- Multi-Seed Aggregated Results for {experiment_name} ---")
        for k, v in aggregated.items():
            print(f"{k}: {v:.4f}")
            
        # Log to file
        output = {
            "experiment": experiment_name,
            "runs": all_metrics,
            "aggregated": aggregated
        }
        
        log_path = os.path.join(self.log_dir, f"{experiment_name}_multiseed.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
            
        print(f"Results logged to {log_path}\n")
        return aggregated
