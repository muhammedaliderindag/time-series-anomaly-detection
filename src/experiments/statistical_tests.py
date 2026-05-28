"""
Statistical Significance Testing Module
=======================================
Runs Wilcoxon signed-rank tests on saved experiment result CSV files.

This module reads:
- results/dl_experiment_results.csv
- logs/skab_automata_multiseed.csv
- logs/batadal_automata_multiseed.csv

It performs paired comparisons where matching runs are available:
- LSTM vs CNN: matched by dataset + seed + fold
- Automata vs DL: matched by dataset + seed

Output files:
- results/statistical_test_results.csv
- results/statistical_test_results.json
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.utils.config_parser import ConfigParser


class StatisticalTester:
    """Performs Wilcoxon signed-rank tests on saved experiment results."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)

        self.results_dir = self.cfg.get("paths.results_dir", "./results")
        self.log_dir = self.cfg.get("paths.log_dir", "./logs")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _safe_read_csv(self, path: str) -> Optional[pd.DataFrame]:
        """Reads a CSV file if it exists."""
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            return None

        return pd.read_csv(path)

    def _read_dl_results(self) -> Optional[pd.DataFrame]:
        """Reads detailed deep learning experiment results."""
        path = os.path.join(self.results_dir, "dl_experiment_results.csv")
        return self._safe_read_csv(path)

    def _read_automata_multiseed(self, dataset: str) -> Optional[pd.DataFrame]:
        """Reads automata multiseed results for a dataset from logs."""
        path = os.path.join(self.log_dir, f"{dataset}_automata_multiseed.csv")
        return self._safe_read_csv(path)

    def _wilcoxon_test(
        self,
        pairs_df: pd.DataFrame,
        value_col_a: str,
        value_col_b: str,
        label_a: str,
        label_b: str,
        dataset: str,
        pairing_key: str
    ) -> Dict[str, object]:
        """Runs a safe Wilcoxon signed-rank test on paired rows."""
        if pairs_df.empty:
            return {
                "dataset": dataset,
                "model_a": label_a,
                "model_b": label_b,
                "pairing_key": pairing_key,
                "n_pairs": 0,
                "statistic": None,
                "p_value": None,
                "significant_at_0_05": None,
                "mean_a": None,
                "mean_b": None,
                "std_a": None,
                "std_b": None,
                "reason": "No matched pairs were available for Wilcoxon test."
            }

        clean_df = pairs_df[[value_col_a, value_col_b]].dropna()
        n_pairs = len(clean_df)

        if n_pairs < 2:
            return {
                "dataset": dataset,
                "model_a": label_a,
                "model_b": label_b,
                "pairing_key": pairing_key,
                "n_pairs": n_pairs,
                "statistic": None,
                "p_value": None,
                "significant_at_0_05": None,
                "mean_a": float(clean_df[value_col_a].mean()) if n_pairs else None,
                "mean_b": float(clean_df[value_col_b].mean()) if n_pairs else None,
                "std_a": float(clean_df[value_col_a].std()) if n_pairs > 1 else None,
                "std_b": float(clean_df[value_col_b].std()) if n_pairs > 1 else None,
                "reason": "Not enough paired samples for Wilcoxon test."
            }

        values_a = clean_df[value_col_a].astype(float).to_numpy()
        values_b = clean_df[value_col_b].astype(float).to_numpy()

        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))
        std_a = float(np.std(values_a, ddof=1))
        std_b = float(np.std(values_b, ddof=1))

        if np.allclose(values_a, values_b):
            return {
                "dataset": dataset,
                "model_a": label_a,
                "model_b": label_b,
                "pairing_key": pairing_key,
                "n_pairs": n_pairs,
                "statistic": 0.0,
                "p_value": 1.0,
                "significant_at_0_05": False,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "std_a": std_a,
                "std_b": std_b,
                "reason": "All paired differences are zero or approximately zero."
            }

        try:
            statistic, p_value = wilcoxon(values_a, values_b)

            return {
                "dataset": dataset,
                "model_a": label_a,
                "model_b": label_b,
                "pairing_key": pairing_key,
                "n_pairs": n_pairs,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant_at_0_05": bool(p_value < 0.05),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "std_a": std_a,
                "std_b": std_b,
                "reason": "Wilcoxon signed-rank test completed."
            }

        except ValueError as exc:
            return {
                "dataset": dataset,
                "model_a": label_a,
                "model_b": label_b,
                "pairing_key": pairing_key,
                "n_pairs": n_pairs,
                "statistic": None,
                "p_value": None,
                "significant_at_0_05": None,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "std_a": std_a,
                "std_b": std_b,
                "reason": f"Wilcoxon test could not be computed: {exc}"
            }

    def _prepare_dl_model_df(
        self,
        dl_df: pd.DataFrame,
        dataset: str,
        model: str
    ) -> pd.DataFrame:
        """Returns DL rows for one dataset/model with standardized columns."""
        required_cols = {"dataset", "model", "seed", "fold", "f1"}

        if not required_cols.issubset(dl_df.columns):
            return pd.DataFrame()

        subset = dl_df[
            (dl_df["dataset"].astype(str).str.lower() == dataset.lower()) &
            (dl_df["model"].astype(str).str.lower() == model.lower())
        ].copy()

        if subset.empty:
            return pd.DataFrame()

        subset["seed"] = subset["seed"].astype(int)
        subset["fold"] = subset["fold"].astype(int)
        subset["f1"] = subset["f1"].astype(float)

        return subset[["dataset", "seed", "fold", "f1"]]

    def _compare_lstm_vs_cnn(
        self,
        dl_df: pd.DataFrame,
        dataset: str
    ) -> Dict[str, object]:
        """Compares LSTM and CNN using matched dataset + seed + fold rows."""
        lstm_df = self._prepare_dl_model_df(dl_df, dataset, "lstm")
        cnn_df = self._prepare_dl_model_df(dl_df, dataset, "cnn")

        if lstm_df.empty or cnn_df.empty:
            return self._wilcoxon_test(
                pd.DataFrame(),
                "f1_lstm",
                "f1_cnn",
                "lstm",
                "cnn",
                dataset,
                "dataset+seed+fold"
            )

        paired = pd.merge(
            lstm_df,
            cnn_df,
            on=["dataset", "seed", "fold"],
            suffixes=("_lstm", "_cnn")
        )

        return self._wilcoxon_test(
            paired,
            "f1_lstm",
            "f1_cnn",
            "lstm",
            "cnn",
            dataset,
            "dataset+seed+fold"
        )

    def _compare_automata_vs_dl(
        self,
        automata_df: pd.DataFrame,
        dl_df: pd.DataFrame,
        dataset: str,
        dl_model: str
    ) -> Dict[str, object]:
        """Compares automata and one DL model using matched dataset + seed rows."""
        required_auto_cols = {"dataset", "seed", "f1"}

        if not required_auto_cols.issubset(automata_df.columns):
            return self._wilcoxon_test(
                pd.DataFrame(),
                "f1_automata",
                f"f1_{dl_model}",
                "automata",
                dl_model,
                dataset,
                "dataset+seed"
            )

        auto = automata_df.copy()
        auto = auto[
            auto["dataset"].astype(str).str.lower() == dataset.lower()
        ].copy()

        if auto.empty:
            return self._wilcoxon_test(
                pd.DataFrame(),
                "f1_automata",
                f"f1_{dl_model}",
                "automata",
                dl_model,
                dataset,
                "dataset+seed"
            )

        auto["seed"] = auto["seed"].astype(int)
        auto["f1"] = auto["f1"].astype(float)

        auto = auto[["dataset", "seed", "f1"]]
        auto = auto.rename(columns={"f1": "f1_automata"})

        dl_model_df = self._prepare_dl_model_df(dl_df, dataset, dl_model)

        if dl_model_df.empty:
            return self._wilcoxon_test(
                pd.DataFrame(),
                "f1_automata",
                f"f1_{dl_model}",
                "automata",
                dl_model,
                dataset,
                "dataset+seed"
            )

        # DL may have multiple folds per seed. Average folds before matching with automata.
        dl_by_seed = (
            dl_model_df
            .groupby(["dataset", "seed"], as_index=False)["f1"]
            .mean()
            .rename(columns={"f1": f"f1_{dl_model}"})
        )

        paired = pd.merge(
            auto,
            dl_by_seed,
            on=["dataset", "seed"]
        )

        return self._wilcoxon_test(
            paired,
            "f1_automata",
            f"f1_{dl_model}",
            "automata",
            dl_model,
            dataset,
            "dataset+seed"
        )

    def run(self) -> List[Dict[str, object]]:
        """Runs available Wilcoxon tests and saves the results."""
        print("--- Starting Statistical Significance Tests ---")

        dl_df = self._read_dl_results()
        datasets = ["skab", "batadal"]

        automata_by_dataset = {
            dataset: self._read_automata_multiseed(dataset)
            for dataset in datasets
        }

        results = []

        if dl_df is not None:
            for dataset in datasets:
                results.append(self._compare_lstm_vs_cnn(dl_df, dataset))

                automata_df = automata_by_dataset.get(dataset)

                if automata_df is not None:
                    results.append(
                        self._compare_automata_vs_dl(
                            automata_df,
                            dl_df,
                            dataset,
                            "cnn"
                        )
                    )
                    results.append(
                        self._compare_automata_vs_dl(
                            automata_df,
                            dl_df,
                            dataset,
                            "lstm"
                        )
                    )

        output_csv = os.path.join(self.results_dir, "statistical_test_results.csv")
        output_json = os.path.join(self.results_dir, "statistical_test_results.json")

        pd.DataFrame(results).to_csv(output_csv, index=False)

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print("\n--- Statistical Test Results ---")
        if results:
            print(pd.DataFrame(results).to_string(index=False))
        else:
            print("No statistical tests were generated.")

        print(f"\nResults saved to {output_csv}")

        return results


if __name__ == "__main__":
    tester = StatisticalTester("configs/config.yaml")
    tester.run()