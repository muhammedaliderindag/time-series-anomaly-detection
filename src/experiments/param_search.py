"""
Automata Parameter Search Module
================================
Runs sensitivity analysis for automata parameters:
- window size: 3, 4, 5, 6
- alphabet size: 3, 4, 5, 6

For each setting, the module reports:
- performance metrics
- number of states
- transition count
- transition density
"""

import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.models.automata.explainability import AutomataExplainability
from src.models.automata.pattern_extractor import PatternExtractor
from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton
from src.models.automata.transforms import SAXTransformer
from src.utils.config_parser import ConfigParser
from src.utils.metrics import calculate_metrics, map_labels_to_patterns


class ParameterSearchTester:
    """Runs parameter variation tests for the probabilistic automata model."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)

        self.log_dir = self.cfg.get("paths.log_dir", "logs")
        self.results_dir = self.cfg.get("paths.results_dir", "results")

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def _evaluate_dir(
        self,
        ds_dir: str,
        window_size: int,
        alphabet_size: int,
        paa_segment_size: int,
        anomaly_threshold: float
    ) -> Dict[str, float]:
        """Evaluates one processed dataset directory for one parameter setting."""
        train_pca = np.load(os.path.join(ds_dir, "train_pca.npy")).flatten()
        test_pca = np.load(os.path.join(ds_dir, "test_pca.npy")).flatten()

        test_label_path = os.path.join(ds_dir, "test_labels.csv")

        if os.path.exists(test_label_path):
            test_lbl = pd.read_csv(test_label_path)["label"].values
        else:
            test_lbl = np.zeros(len(test_pca))

        sax = SAXTransformer(
            segment_size=paa_segment_size,
            alphabet_size=alphabet_size
        )
        extractor = PatternExtractor(window_size=window_size)

        train_patterns = extractor.extract_patterns(sax.transform(train_pca))
        test_patterns = extractor.extract_patterns(sax.transform(test_pca))

        model = ProbabilisticAutomaton(
            model_dir=self.cfg.get("paths.model_dir", "./models")
        )
        model.fit(train_patterns)

        explainability = AutomataExplainability(model, anomaly_threshold)
        test_justifications, path_probability, confidence_score = explainability.explain_path(
            test_patterns
        )

        test_labels = map_labels_to_patterns(
            test_lbl,
            paa_segment_size,
            window_size
        )
        test_preds = np.array([
            1 if item["decision"] == "anomaly" else 0
            for item in test_justifications
        ])

        metrics = calculate_metrics(test_labels, test_preds)

        return {
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1_score": float(metrics.get("f1", 0.0)),
            "state_count": float(model.get_state_count()),
            "transition_count": float(model.get_transition_count()),
            "transition_density": float(model.get_transition_density()),
            "path_probability": float(path_probability),
            "confidence_score": float(confidence_score)
        }

    def _average_fold_results(self, fold_results: list) -> Dict[str, float]:
        """Averages numeric metrics across SKAB folds."""
        if not fold_results:
            return {}

        keys = fold_results[0].keys()
        averaged = {}

        for key in keys:
            values = [float(result[key]) for result in fold_results]
            averaged[key] = float(np.mean(values))
            averaged[f"{key}_std"] = float(np.std(values))

        return averaged

    def run_grid_search(self) -> None:
        """Runs the full automata parameter grid search."""
        print("--- Starting Automata Parameter Variation Testing ---")

        window_sizes = [3, 4, 5, 6]
        alphabet_sizes = [3, 4, 5, 6]

        paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)
        data_dir = self.cfg.get("paths.data_dir")

        datasets = ["skab", "batadal"]
        all_results = []

        for dataset_name in datasets:
            print(f"\n[Grid Search on Dataset]: {dataset_name.upper()}")

            processed_dir = os.path.join(data_dir, "processed", dataset_name)

            for window_size in window_sizes:
                for alphabet_size in alphabet_sizes:
                    print(
                        f"  Testing Window={window_size}, "
                        f"Alphabet={alphabet_size}..."
                    )

                    if dataset_name == "skab":
                        fold_results = []
                        fold_dirs = sorted([
                            item for item in os.listdir(processed_dir)
                            if item.startswith("fold_")
                        ])

                        for fold_dir in fold_dirs:
                            fold_path = os.path.join(processed_dir, fold_dir)
                            fold_results.append(
                                self._evaluate_dir(
                                    fold_path,
                                    window_size,
                                    alphabet_size,
                                    paa_segment_size,
                                    anomaly_threshold
                                )
                            )

                        averaged_metrics = self._average_fold_results(fold_results)

                        all_results.append({
                            "dataset": dataset_name,
                            "window_size": window_size,
                            "alphabet_size": alphabet_size,
                            **averaged_metrics
                        })

                    else:
                        metrics = self._evaluate_dir(
                            processed_dir,
                            window_size,
                            alphabet_size,
                            paa_segment_size,
                            anomaly_threshold
                        )

                        all_results.append({
                            "dataset": dataset_name,
                            "window_size": window_size,
                            "alphabet_size": alphabet_size,
                            **metrics
                        })

        for output_dir in [self.log_dir, self.results_dir]:
            json_path = os.path.join(output_dir, "automata_param_search.json")
            csv_path = os.path.join(output_dir, "automata_param_search.csv")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)

            pd.DataFrame(all_results).to_csv(csv_path, index=False)

        print(f"\nResults saved to {self.results_dir}/automata_param_search.csv")