"""
Unseen Pattern Analysis Module
==============================
Aggregates seen/unseen pattern statistics from automata explainability JSON files.

Outputs:
- results/unseen_analysis_results.csv
- results/unseen_analysis_results.json
"""

import json
import os
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.config_parser import ConfigParser


class UnseenAnalysisRunner:
    """Creates report-level unseen pattern statistics from explainability outputs."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)
        self.results_dir = self.cfg.get("paths.results_dir", "./results")
        os.makedirs(self.results_dir, exist_ok=True)

    def _load_json(self, path: str) -> Dict[str, Any]:
        """Loads one explainability JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _parse_dataset_and_fold(self, filename: str) -> Dict[str, Any]:
        """
        Parses dataset and fold index from filenames like:
        - skab_fold0_explainability.json
        - batadal_fold0_explainability.json
        """
        match = re.match(r"(?P<dataset>.+)_fold(?P<fold>\d+)_explainability\.json", filename)

        if not match:
            return {
                "dataset": "unknown",
                "fold": -1
            }

        return {
            "dataset": match.group("dataset"),
            "fold": int(match.group("fold"))
        }

    def _extract_steps(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts step-level justifications from different possible JSON layouts.
        """
        if "justifications" in payload and isinstance(payload["justifications"], list):
            return payload["justifications"]

        if "steps" in payload and isinstance(payload["steps"], list):
            return payload["steps"]

        if "path" in payload and isinstance(payload["path"], list):
            return payload["path"]

        if isinstance(payload, list):
            return payload

        return []

    def _summarize_file(self, path: str) -> Dict[str, Any]:
        """Summarizes unseen/seen behavior for one explainability file."""
        filename = os.path.basename(path)
        parsed = self._parse_dataset_and_fold(filename)

        payload = self._load_json(path)
        steps = self._extract_steps(payload)

        total_steps = len(steps)

        seen_steps = [
            step for step in steps
            if str(step.get("status", "")).lower() == "seen"
        ]

        unseen_steps = [
            step for step in steps
            if str(step.get("status", "")).lower() == "unseen"
        ]

        mapped_unseen_steps = [
            step for step in unseen_steps
            if step.get("mapped_to") is not None
        ]

        edit_distances = [
            step.get("edit_distance")
            for step in unseen_steps
            if step.get("edit_distance") is not None
        ]

        anomaly_steps = [
            step for step in steps
            if str(step.get("decision", "")).lower() == "anomaly"
        ]

        unseen_anomaly_steps = [
            step for step in unseen_steps
            if str(step.get("decision", "")).lower() == "anomaly"
        ]

        transition_probs = []
        for step in steps:
            transition = step.get("transition")
            if isinstance(transition, dict) and transition.get("probability") is not None:
                transition_probs.append(float(transition["probability"]))
            elif step.get("probability") is not None:
                transition_probs.append(float(step["probability"]))

        seen_count = len(seen_steps)
        unseen_count = len(unseen_steps)
        mapped_unseen_count = len(mapped_unseen_steps)
        anomaly_count = len(anomaly_steps)
        unseen_anomaly_count = len(unseen_anomaly_steps)

        return {
            "dataset": parsed["dataset"],
            "fold": parsed["fold"],
            "file": filename,
            "total_steps": total_steps,
            "seen_count": seen_count,
            "unseen_count": unseen_count,
            "unseen_rate": float(unseen_count / total_steps) if total_steps else 0.0,
            "mapped_unseen_count": mapped_unseen_count,
            "mapped_unseen_rate": float(mapped_unseen_count / unseen_count) if unseen_count else 0.0,
            "mean_edit_distance": float(np.mean(edit_distances)) if edit_distances else 0.0,
            "max_edit_distance": float(np.max(edit_distances)) if edit_distances else 0.0,
            "anomaly_count": anomaly_count,
            "anomaly_rate": float(anomaly_count / total_steps) if total_steps else 0.0,
            "unseen_anomaly_count": unseen_anomaly_count,
            "unseen_anomaly_rate": float(unseen_anomaly_count / unseen_count) if unseen_count else 0.0,
            "mean_transition_probability": float(np.mean(transition_probs)) if transition_probs else 0.0,
            "min_transition_probability": float(np.min(transition_probs)) if transition_probs else 0.0,
            "zero_probability_transition_count": int(sum(prob == 0.0 for prob in transition_probs))
        }

    def run(self) -> pd.DataFrame:
        """Runs unseen pattern analysis and saves CSV/JSON outputs."""
        print("--- Starting Unseen Pattern Analysis ---")

        explainability_files = sorted([
            os.path.join(self.results_dir, filename)
            for filename in os.listdir(self.results_dir)
            if filename.endswith("_explainability.json")
        ])

        if not explainability_files:
            print("No explainability JSON files found.")
            return pd.DataFrame()

        rows = [
            self._summarize_file(path)
            for path in explainability_files
        ]

        df = pd.DataFrame(rows)

        csv_path = os.path.join(self.results_dir, "unseen_analysis_results.csv")
        json_path = os.path.join(self.results_dir, "unseen_analysis_results.json")

        df.to_csv(csv_path, index=False)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=4, ensure_ascii=False)

        print("\n--- Unseen Pattern Analysis Results ---")
        print(df.to_string(index=False))
        print(f"\nResults saved to {csv_path}")

        return df


if __name__ == "__main__":
    runner = UnseenAnalysisRunner("configs/config.yaml")
    runner.run()