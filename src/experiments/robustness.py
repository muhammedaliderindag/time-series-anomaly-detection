"""
Robustness Testing Module
=========================
Evaluates the automata model under Gaussian noise.

This module measures how the probabilistic automaton behaves when noise is
injected into the test PCA signal. It reports original and noisy performance
for SKAB and BATADAL.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models.automata.explainability import AutomataExplainability
from src.models.automata.pattern_extractor import PatternExtractor
from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton
from src.models.automata.transforms import SAXTransformer
from src.utils.config_parser import ConfigParser
from src.utils.metrics import calculate_metrics, map_labels_to_patterns


class RobustnessTester:
    """Runs Gaussian noise robustness tests for the automata model."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)

        self.log_dir = self.cfg.get("paths.log_dir", "logs")
        self.results_dir = self.cfg.get("paths.results_dir", "results")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def inject_gaussian_noise(
        self,
        data: np.ndarray,
        mean: float = 0.0,
        std: float = 0.1,
        seed: int = 42
    ) -> np.ndarray:
        """Adds reproducible Gaussian noise to the given PCA signal."""
        rng = np.random.default_rng(seed)
        noise = rng.normal(mean, std, size=data.shape)
        return data + noise

    def evaluate_automata(
        self,
        dataset_name: str,
        test_pca: np.ndarray,
        test_lbl_orig: np.ndarray,
        fold_idx: int = 0
    ) -> Dict[str, float]:
        """Evaluates the trained automata model on a test signal."""
        paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        alphabet_size = self.cfg.get("automata.alphabet_size", 3)
        window_size = self.cfg.get("automata.window_size", 4)
        anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)

        model_dir = self.cfg.get("paths.model_dir")
        model = ProbabilisticAutomaton(model_dir=model_dir)

        model_name = (
            f"{dataset_name}_fold{fold_idx}_automaton.json"
            if dataset_name == "skab"
            else f"{dataset_name}_fold0_automaton.json"
        )

        model_path = os.path.join(model.artifact_dir, model_name)

        if not os.path.exists(model_path):
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

        model.load(model_path)

        sax = SAXTransformer(
            segment_size=paa_segment_size,
            alphabet_size=alphabet_size
        )
        extractor = PatternExtractor(window_size=window_size)

        test_patterns = extractor.extract_patterns(sax.transform(test_pca))

        explainability = AutomataExplainability(model, anomaly_threshold)
        test_justifications, _, _ = explainability.explain_path(test_patterns)

        test_labels = map_labels_to_patterns(
            test_lbl_orig,
            paa_segment_size,
            window_size
        )

        test_preds = np.array([
            1 if item["decision"] == "anomaly" else 0
            for item in test_justifications
        ])

        return calculate_metrics(test_labels, test_preds)

    def _get_test_data_for_ds(
        self,
        dataset_name: str,
        data_dir: str
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Loads processed test PCA arrays and labels for a dataset."""
        processed_dir = os.path.join(data_dir, "processed", dataset_name)

        if dataset_name == "skab":
            test_pcas = []
            test_labels = []

            fold_dirs = sorted([
                item for item in os.listdir(processed_dir)
                if item.startswith("fold_")
            ])

            for fold_dir in fold_dirs:
                fold_path = os.path.join(processed_dir, fold_dir)

                test_pca_path = os.path.join(fold_path, "test_pca.npy")
                test_label_path = os.path.join(fold_path, "test_labels.csv")

                if not os.path.exists(test_pca_path):
                    continue

                test_pca = np.load(test_pca_path).flatten()
                test_pcas.append(test_pca)

                if os.path.exists(test_label_path):
                    test_labels.append(pd.read_csv(test_label_path)["label"].values)
                else:
                    test_labels.append(np.zeros(len(test_pca)))

            return test_pcas, test_labels

        test_pca_path = os.path.join(processed_dir, "test_pca.npy")
        test_label_path = os.path.join(processed_dir, "test_labels.csv")

        if not os.path.exists(test_pca_path):
            return [], []

        test_pca = np.load(test_pca_path).flatten()

        if os.path.exists(test_label_path):
            test_label = pd.read_csv(test_label_path)["label"].values
        else:
            test_label = np.zeros(len(test_pca))

        return [test_pca], [test_label]

    def _mean_metric(
        self,
        metric_list: List[Dict[str, float]],
        metric_name: str
    ) -> float:
        """Computes the mean of one metric across folds."""
        if not metric_list:
            return 0.0

        return float(np.mean([metrics.get(metric_name, 0.0) for metrics in metric_list]))

    def run_robustness_test(self, std_list: list = None) -> None:
        """Runs Gaussian noise robustness tests for all project datasets."""
        if std_list is None:
            std_list = [0.05, 0.1, 0.2, 0.5]

        print("--- Starting Automata Robustness Testing (Gaussian Noise) ---")

        datasets = ["skab", "batadal"]
        results = []
        data_dir = self.cfg.get("paths.data_dir")

        for dataset_name in datasets:
            print(f"Testing robustness for dataset: {dataset_name}")

            test_pcas, test_labels = self._get_test_data_for_ds(dataset_name, data_dir)

            if not test_pcas:
                continue

            original_metrics = []

            for fold_idx, (test_pca, test_lbl_orig) in enumerate(zip(test_pcas, test_labels)):
                original_metrics.append(
                    self.evaluate_automata(
                        dataset_name,
                        test_pca,
                        test_lbl_orig,
                        fold_idx
                    )
                )

            result_row = {
                "dataset": dataset_name,
                "automata_original_accuracy": self._mean_metric(original_metrics, "accuracy"),
                "automata_original_precision": self._mean_metric(original_metrics, "precision"),
                "automata_original_recall": self._mean_metric(original_metrics, "recall"),
                "automata_original_f1": self._mean_metric(original_metrics, "f1")
            }

            for std in std_list:
                noisy_metrics = []

                for fold_idx, (test_pca, test_lbl_orig) in enumerate(zip(test_pcas, test_labels)):
                    noise_seed = 42 + fold_idx + int(std * 1000)
                    noisy_test_pca = self.inject_gaussian_noise(
                        test_pca,
                        std=std,
                        seed=noise_seed
                    )
                    

                    noisy_metrics.append(
                        self.evaluate_automata(
                            dataset_name,
                            noisy_test_pca,
                            test_lbl_orig,
                            fold_idx
                        )
                    )

                result_row[f"automata_noise_{std}_accuracy"] = self._mean_metric(
                    noisy_metrics,
                    "accuracy"
                )
                result_row[f"automata_noise_{std}_precision"] = self._mean_metric(
                    noisy_metrics,
                    "precision"
                )
                result_row[f"automata_noise_{std}_recall"] = self._mean_metric(
                    noisy_metrics,
                    "recall"
                )
                result_row[f"automata_noise_{std}_f1"] = self._mean_metric(
                    noisy_metrics,
                    "f1"
                )

            results.append(result_row)

        df = pd.DataFrame(results)

        for output_dir in [self.log_dir, self.results_dir]:
            json_path = os.path.join(output_dir, "robustness_test_results.json")
            csv_path = os.path.join(output_dir, "robustness_test_results.csv")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

            df.to_csv(csv_path, index=False)

        print("\n--- Robustness Results ---")
        print(df.to_string(index=False))
        print(f"\nResults saved to {self.results_dir}/robustness_test_results.csv")