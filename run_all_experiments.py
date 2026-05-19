"""
Master Execution Script for Experimental Scenarios
==================================================
Runs all experimental automation:
1. Multi-seed execution
2. Robustness testing (Gaussian Noise)
3. Cross-Dataset Generalization
4. Automata Parameter Variation
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments.runner import MultiSeedRunner
from src.experiments.robustness import RobustnessTester
from src.experiments.cross_dataset import CrossDatasetTester
from src.experiments.param_search import ParameterSearchTester

def run_pipeline_for_seed(seed: int) -> dict:
    """Wrapper function for multi-seed execution."""
    from src.pipelines.anomaly_detection_pipeline import AnomalyDetectionPipeline
    import numpy as np
    import torch
    import random
    
    # Enforce reproducibility for this run
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # We will test on SWAT for multi-seed as a representative example
    pipeline = AnomalyDetectionPipeline("configs/config.yaml")
    res = pipeline.run_single_dataset("swat")
    return res["test_metrics"]

def main():
    print("=" * 60)
    print("STARTING PHASE 5: EXPERIMENTAL AUTOMATION & SCENARIO TESTING")
    print("=" * 60)
    
    # Step 1: Multi-Seed Execution
    print("\n>>> STEP 1: Multi-Seed Execution Wrapper")
    runner = MultiSeedRunner("configs/config.yaml")
    runner.run("swat_automata_multiseed", run_pipeline_for_seed)
    
    # Step 2: Noise Injection & Robustness Testing
    print("\n>>> STEP 2: Noise Injection & Robustness Testing")
    robustness = RobustnessTester("configs/config.yaml")
    robustness.run_robustness_test(std_list=[0.05, 0.1, 0.2, 0.5])
    
    # Step 3: Cross-Dataset Generalization Testing
    print("\n>>> STEP 3: Cross-Dataset Generalization Testing")
    cross_tester = CrossDatasetTester("configs/config.yaml")
    cross_tester.evaluate_cross_dataset()
    
    # Step 4: Automata Parameter Variation Testing
    print("\n>>> STEP 4: Automata Parameter Variation Testing")
    param_search = ParameterSearchTester("configs/config.yaml")
    param_search.run_grid_search()
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("Check the 'logs/' directory for the aggregated CSV and JSON results.")
    print("=" * 60)

if __name__ == "__main__":
    main()
