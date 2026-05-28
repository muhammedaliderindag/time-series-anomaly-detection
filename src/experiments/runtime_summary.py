"""
Runtime Summary Module
======================
Creates a report-level runtime comparison for Automata and Deep Learning models.

Outputs:
- results/runtime_summary.csv
- results/runtime_summary.json
"""

import json
import os
import time
from typing import Any, Dict, List

import pandas as pd

from src.pipelines.anomaly_detection_pipeline import AnomalyDetectionPipeline
from src.utils.config_parser import ConfigParser


class RuntimeSummaryRunner:
    """Generates training/inference runtime summary tables."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)
        self.results_dir = self.cfg.get("paths.results_dir", "./results")
        os.makedirs(self.results_dir, exist_ok=True)

    def _measure_automata_runtime(self, dataset_name: str) -> Dict[str, Any]:
        """
        Measures end-to-end automata runtime for one dataset.

        The automata pipeline is very fast, so this reports total pipeline time
        as the automata runtime proxy.
        """
        pipeline = AnomalyDetectionPipeline("configs/config.yaml")

        start = time.perf_counter()
        result = pipeline.run_single_dataset(dataset_name)
        total_time = time.perf_counter() - start

        metrics = result.get("test_metrics", {})

        return {
            "dataset": dataset_name,
            "model": "automata",
            "training_time_sec_mean": float(total_time),
            "training_time_sec_std": 0.0,
            "inference_time_sec_mean": 0.0,
            "inference_time_sec_std": 0.0,
            "accuracy_mean": float(metrics.get("accuracy", 0.0)),
            "f1_mean": float(metrics.get("f1", 0.0)),
            "source": "measured_end_to_end_pipeline_time"
        }

    def _load_dl_runtime_rows(self) -> List[Dict[str, Any]]:
        """Loads DL runtime summary rows if available."""
        summary_path = os.path.join(self.results_dir, "dl_experiment_summary.csv")

        if not os.path.exists(summary_path):
            return []

        df = pd.read_csv(summary_path)
        rows = []

        for _, row in df.iterrows():
            rows.append({
                "dataset": row.get("dataset"),
                "model": row.get("model"),
                "training_time_sec_mean": float(row.get("training_time_sec_mean", 0.0)),
                "training_time_sec_std": float(row.get("training_time_sec_std", 0.0)),
                "inference_time_sec_mean": float(row.get("inference_time_sec_mean", 0.0)),
                "inference_time_sec_std": float(row.get("inference_time_sec_std", 0.0)),
                "accuracy_mean": float(row.get("accuracy_mean", 0.0)),
                "f1_mean": float(row.get("f1_mean", 0.0)),
                "source": "dl_experiment_summary"
            })

        return rows

    def run(self) -> pd.DataFrame:
        """Creates runtime summary CSV/JSON."""
        print("--- Starting Runtime Summary Generation ---")

        rows = []

        for dataset_name in ["skab", "batadal"]:
            rows.append(self._measure_automata_runtime(dataset_name))

        rows.extend(self._load_dl_runtime_rows())

        df = pd.DataFrame(rows)

        csv_path = os.path.join(self.results_dir, "runtime_summary.csv")
        json_path = os.path.join(self.results_dir, "runtime_summary.json")

        df.to_csv(csv_path, index=False)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=4, ensure_ascii=False)

        print("\n--- Runtime Summary ---")
        print(df.to_string(index=False))
        print(f"\nResults saved to {csv_path}")

        return df


if __name__ == "__main__":
    runner = RuntimeSummaryRunner("configs/config.yaml")
    runner.run()