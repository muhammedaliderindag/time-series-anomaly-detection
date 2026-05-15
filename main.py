"""
Main Pipeline Entrypoint
=========================
Serves as the clean, professional z-normalized ML orchestrator execution entry point.
Invokes the preprocessing pipeline and runs anomaly detection model pipelines
across SWAT, WADI, and BATADAL datasets.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipelines.anomaly_detection_pipeline import AnomalyDetectionPipeline


def main():
    print("=" * 60)
    print("STARTING TIME SERIES ANOMALY DETECTION PROJECT")
    print("=" * 60)

    # 1. Initialize Pipeline (handles config loading & logging setup internally)
    pipeline = AnomalyDetectionPipeline(config_path="configs/config.yaml")

    # 2. Run Data Preprocessing
    print("\nRunning Feature Engineering and Preprocessing...")
    pipeline.run_preprocessing()

    # 3. Train & Evaluate Models across all datasets
    print("\nRunning Model Training and Evaluation...")
    results = pipeline.run_all()

    # 4. Final verification output
    print("\n" + "=" * 60)
    print("EXPERIMENT EXECUTION COMPLETE")
    print("=" * 60)
    for res in results:
        test_metrics = res["test_metrics"]
        print(f"Dataset: {res['dataset'].upper():<10} | "
              f"Test F1-Score: {test_metrics['f1']:.4f} | "
              f"Accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
