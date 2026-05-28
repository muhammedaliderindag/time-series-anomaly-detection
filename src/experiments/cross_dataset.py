"""
Cross-Dataset Generalization Module
===================================
Evaluates how an automata model trained on one dataset generalizes to another.

For this project, the required datasets are:
- SKAB
- BATADAL

The output is saved as both JSON and CSV for report tables.
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


class CrossDatasetTester:
    """Runs cross-dataset generalization tests for the automata model."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)

        self.log_dir = self.cfg.get("paths.log_dir", "logs")
        self.results_dir = self.cfg.get("paths.results_dir", "results")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.datasets = ["skab", "batadal"]

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

    def _load_source_model(
        self,
        dataset_name: str,
        fold_idx: int = 0
    ) -> ProbabilisticAutomaton:
        """Loads a trained automata model for the source dataset."""
        model_dir = self.cfg.get("paths.model_dir")
        model = ProbabilisticAutomaton(model_dir=model_dir)

        model_name = f"{dataset_name}_fold{fold_idx}_automaton.json"
        model_path = os.path.join(model.artifact_dir, model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        model.load(model_path)
        return model

    def _evaluate_model_on_dataset(
        self,
        model: ProbabilisticAutomaton,
        dataset_name: str,
        test_pcas: List[np.ndarray],
        test_labels: List[np.ndarray]
    ) -> Dict[str, float]:
        """Evaluates one source model on all folds/items of a target dataset."""
        paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        alphabet_size = self.cfg.get("automata.alphabet_size", 3)
        window_size = self.cfg.get("automata.window_size", 4)
        anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)

        fold_metrics = []

        for test_pca, test_lbl_orig in zip(test_pcas, test_labels):
            sax = SAXTransformer(
                segment_size=paa_segment_size,
                alphabet_size=alphabet_size
            )
            extractor = PatternExtractor(window_size=window_size)

            test_patterns = extractor.extract_patterns(sax.transform(test_pca))

            explainability = AutomataExplainability(model, anomaly_threshold)
            test_justifications, _, _ = explainability.explain_path(test_patterns)

            y_true = map_labels_to_patterns(
                test_lbl_orig,
                paa_segment_size,
                window_size
            )
            y_pred = np.array([
                1 if item["decision"] == "anomaly" else 0
                for item in test_justifications
            ])

            fold_metrics.append(calculate_metrics(y_true, y_pred))

        if not fold_metrics:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

        return {
            "accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "precision": float(np.mean([m["precision"] for m in fold_metrics])),
            "recall": float(np.mean([m["recall"] for m in fold_metrics])),
            "f1": float(np.mean([m["f1"] for m in fold_metrics]))
        }

    def evaluate_cross_dataset(self) -> None:
        """Runs full SKAB/BATADAL cross-dataset evaluation."""
        print("--- Starting Cross-Dataset Generalization Testing ---")

        data_dir = self.cfg.get("paths.data_dir")
        detailed_results = []
        f1_matrix_rows = []

        for train_dataset in self.datasets:
            print(f"\n[Source Model]: {train_dataset.upper()}")

            try:
                model = self._load_source_model(train_dataset, fold_idx=0)
            except FileNotFoundError as exc:
                print(str(exc))
                continue

            matrix_row = {"train_dataset": train_dataset}

            for test_dataset in self.datasets:
                test_pcas, test_labels = self._get_test_data_for_ds(
                    test_dataset,
                    data_dir
                )

                if not test_pcas:
                    matrix_row[test_dataset] = None
                    continue

                metrics = self._evaluate_model_on_dataset(
                    model,
                    test_dataset,
                    test_pcas,
                    test_labels
                )

                detailed_results.append({
                    "train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                    **metrics
                })

                matrix_row[test_dataset] = metrics["f1"]

                print(
                    f"  -> Test on {test_dataset.upper()}: "
                    f"Accuracy={metrics['accuracy']:.4f}, "
                    f"F1={metrics['f1']:.4f}"
                )

            f1_matrix_rows.append(matrix_row)

        detailed_df = pd.DataFrame(detailed_results)
        matrix_df = pd.DataFrame(f1_matrix_rows)

        for output_dir in [self.log_dir, self.results_dir]:
            detailed_json_path = os.path.join(
                output_dir,
                "cross_dataset_results.json"
            )
            detailed_csv_path = os.path.join(
                output_dir,
                "cross_dataset_results.csv"
            )
            matrix_csv_path = os.path.join(
                output_dir,
                "cross_dataset_matrix.csv"
            )

            with open(detailed_json_path, "w", encoding="utf-8") as f:
                json.dump(detailed_results, f, indent=4, ensure_ascii=False)

            detailed_df.to_csv(detailed_csv_path, index=False)
            matrix_df.to_csv(matrix_csv_path, index=False)

        print("\n--- Cross-Dataset Matrix (F1-Scores) ---")
        print(matrix_df.to_string(index=False))
        print(f"\nResults saved to {self.results_dir}/cross_dataset_matrix.csv")