"""
Master Execution Script for Experimental Scenarios
==================================================
Runs experimental automation:
1. Multi-seed execution for SKAB and BATADAL automata experiments
2. Robustness testing with Gaussian noise
3. Cross-dataset generalization
4. Automata parameter variation
5. Optional deep learning experiments
"""

import argparse
import os
import random
import sys
from typing import Dict

import numpy as np
import torch
from src.experiments.statistical_tests import StatisticalTester
# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments.cross_dataset import CrossDatasetTester
from src.experiments.dl_experiments import DeepLearningExperimentRunner
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


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run automata and optional deep learning experiments."
    )

    parser.add_argument(
        "--include-dl",
        action="store_true",
        help="Also run LSTM and 1D-CNN deep learning experiments. This can take a long time."
    )

    parser.add_argument(
        "--dl-smoke-test",
        action="store_true",
        help="Run a quick DL smoke test instead of the full DL experiment."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("STARTING EXPERIMENTAL AUTOMATION & SCENARIO TESTING")
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

    if args.dl_smoke_test:
        print("\n>>> STEP 5: Deep Learning Smoke Test")
        dl_runner = DeepLearningExperimentRunner("configs/config.yaml")
        dl_runner.run(
            datasets=["batadal"],
            models=["cnn"],
            seeds=[42]
        )

    elif args.include_dl:
        print("\n>>> STEP 5: Full Deep Learning Experiments")
        print("WARNING: This may take a long time on CPU.")
        dl_runner = DeepLearningExperimentRunner("configs/config.yaml")
        dl_runner.run()

    else:
        print("\n>>> STEP 5: Deep Learning Experiments Skipped")
        print("Use --dl-smoke-test for a quick check or --include-dl for full DL experiments.")

    print("\n>>> STEP 6: Statistical Significance Testing")
    statistical_tester = StatisticalTester("configs/config.yaml")
    statistical_tester.run()

    print("\n" + "=" * 60)
    print("ALL REQUESTED EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("Check the 'logs/' and 'results/' directories for experiment outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()