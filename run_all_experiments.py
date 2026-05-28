"""
Master Execution Script for Experimental Scenarios
==================================================
Runs all experimental automation:
1. Multi-seed execution for SKAB and BATADAL
2. Robustness testing with Gaussian noise
3. Cross-dataset generalization
4. Automata parameter variation
"""

import os
import random
import sys
from typing import Dict

import numpy as np
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments.cross_dataset import CrossDatasetTester
from src.experiments.param_search import ParameterSearchTester
from src.experiments.robustness import RobustnessTester
from src.experiments.runner import MultiSeedRunner
from src.pipelines.anomaly_detection_pipeline import AnomalyDetectionPipeline


def set_global_seed(seed: int) -> None:
    """Sets random seeds for reproducible experiment execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline_for_dataset_and_seed(dataset_name: str, seed: int) -> Dict[str, float]:
    """Runs the automata pipeline for one dataset and one random seed."""
    set_global_seed(seed)

    pipeline = AnomalyDetectionPipeline("configs/config.yaml")
    result = pipeline.run_single_dataset(dataset_name)

    metrics = result["test_metrics"].copy()
    metrics["dataset"] = dataset_name

    return metrics


def main():
    print("=" * 60)
    print("STARTING PHASE 5: EXPERIMENTAL AUTOMATION & SCENARIO TESTING")
    print("=" * 60)

    runner = MultiSeedRunner("configs/config.yaml")

    print("\n>>> STEP 1: Multi-Seed Execution for SKAB and BATADAL")
    for dataset_name in ["skab", "batadal"]:
        runner.run(
            f"{dataset_name}_automata_multiseed",
            lambda seed, ds=dataset_name: run_pipeline_for_dataset_and_seed(ds, seed)
        )

    print("\n>>> STEP 2: Noise Injection & Robustness Testing")
    robustness = RobustnessTester("configs/config.yaml")
    robustness.run_robustness_test(std_list=[0.05, 0.1, 0.2, 0.5])

    print("\n>>> STEP 3: Cross-Dataset Generalization Testing")
    cross_tester = CrossDatasetTester("configs/config.yaml")
    cross_tester.evaluate_cross_dataset()

    print("\n>>> STEP 4: Automata Parameter Variation Testing")
    param_search = ParameterSearchTester("configs/config.yaml")
    param_search.run_grid_search()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("Check the 'logs/' and 'results/' directories for experiment outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()