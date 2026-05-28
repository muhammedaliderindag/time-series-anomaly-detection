"""
Anomaly Detection Pipeline Orchestrator
=======================================
Orchestrates model training, validation, testing, evaluation,
and explainability outputs for the Time Series Anomaly Detection project.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.data.build_features import run_pipeline
from src.models.automata.explainability import AutomataExplainability
from src.models.automata.pattern_extractor import PatternExtractor
from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton
from src.models.automata.transforms import SAXTransformer
from src.utils.config_parser import ConfigParser
from src.utils.logger import ExperimentLogger
from src.utils.metrics import calculate_metrics, map_labels_to_patterns


class AnomalyDetectionPipeline:
    """Manages automata-based anomaly detection pipelines across datasets."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.cfg = ConfigParser(config_path)
        self.logger = ExperimentLogger(self.cfg.config, experiment_name="automata_experiment")

        self.anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)
        self.paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        self.alphabet_size = self.cfg.get("automata.alphabet_size", 3)
        self.window_size = self.cfg.get("automata.window_size", 4)

        self.results_dir = self.cfg.get("paths.results_dir", "./results")
        os.makedirs(self.results_dir, exist_ok=True)

    def run_preprocessing(self) -> None:
        """Runs the preprocessing and feature engineering pipeline."""
        self.logger.info("Starting preprocessing step...")
        run_pipeline(self.config_path)
        self.logger.info("Preprocessing complete.")

    def _save_json(self, payload: Dict[str, Any], filename: str) -> str:
        """Saves a JSON payload under the configured results directory."""
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return filepath

    def _save_explainability_output(
        self,
        dataset_name: str,
        fold_idx: int,
        explainability: AutomataExplainability,
        justifications: List[Dict[str, Any]],
        path_probability: float,
        confidence_score: float,
        max_steps: int = 25
    ) -> str:
        """
        Saves report-ready explainability output.

        Only the first max_steps are saved to keep the JSON readable for reports.
        The full decision logic is still computed on the complete test path.
        """
        payload = {
            "dataset": dataset_name,
            "fold": fold_idx,
            "automata_parameters": {
                "paa_segment_size": self.paa_segment_size,
                "alphabet_size": self.alphabet_size,
                "window_size": self.window_size,
                "anomaly_threshold": self.anomaly_threshold
            },
            "summary": explainability.build_summary(
                justifications,
                path_probability,
                confidence_score
            ),
            "steps": justifications[:max_steps]
        }

        filename = f"{dataset_name}_fold{fold_idx}_explainability.json"
        return self._save_json(payload, filename)

    def _load_labels_or_default(self, filepath: str, length: int) -> np.ndarray:
        """Loads labels if available; otherwise returns a zero-label vector."""
        if os.path.exists(filepath):
            return pd.read_csv(filepath)["label"].values

        return np.zeros(length)

    def _run_fold(self, dataset_name: str, fold_dir: str, fold_idx: int) -> Dict[str, Any]:
        """Runs training and evaluation for a single dataset fold."""
        train_pca = np.load(os.path.join(fold_dir, "train_pca.npy")).flatten()
        val_pca = np.load(os.path.join(fold_dir, "val_pca.npy")).flatten()
        test_pca = np.load(os.path.join(fold_dir, "test_pca.npy")).flatten()

        train_lbl = self._load_labels_or_default(
            os.path.join(fold_dir, "train_labels.csv"),
            len(train_pca)
        )
        val_lbl = self._load_labels_or_default(
            os.path.join(fold_dir, "val_labels.csv"),
            len(val_pca)
        )
        test_lbl = self._load_labels_or_default(
            os.path.join(fold_dir, "test_labels.csv"),
            len(test_pca)
        )

        sax = SAXTransformer(
            segment_size=self.paa_segment_size,
            alphabet_size=self.alphabet_size
        )
        extractor = PatternExtractor(window_size=self.window_size)

        train_patterns = extractor.extract_patterns(sax.transform(train_pca))
        val_patterns = extractor.extract_patterns(sax.transform(val_pca))
        test_patterns = extractor.extract_patterns(sax.transform(test_pca))

        model = ProbabilisticAutomaton(model_dir=self.cfg.get("paths.model_dir"))
        model.fit(train_patterns)
        model.save(f"{dataset_name}_fold{fold_idx}_automaton.json")

        explainability = AutomataExplainability(model, self.anomaly_threshold)

        test_justifications, path_probability, confidence_score = explainability.explain_path(
            test_patterns
        )

        explanation_path = self._save_explainability_output(
            dataset_name=dataset_name,
            fold_idx=fold_idx,
            explainability=explainability,
            justifications=test_justifications,
            path_probability=path_probability,
            confidence_score=confidence_score
        )

        test_labels = map_labels_to_patterns(
            test_lbl,
            self.paa_segment_size,
            self.window_size
        )
        test_preds = np.array([
            1 if item["decision"] == "anomaly" else 0
            for item in test_justifications
        ])

        test_metrics = calculate_metrics(test_labels, test_preds)

        test_metrics.update({
            "state_count": model.get_state_count(),
            "transition_count": model.get_transition_count(),
            "transition_density": model.get_transition_density(),
            "path_probability": float(path_probability),
            "confidence_score": float(confidence_score),
            "explainability_file": explanation_path
        })

        return test_metrics

    def _summarize_fold_metrics(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes mean and standard deviation for numeric fold metrics."""
        summary: Dict[str, Any] = {}

        metric_keys = [
            key for key, value in fold_metrics[0].items()
            if isinstance(value, (int, float, np.integer, np.floating))
        ]

        for key in metric_keys:
            values = np.array([float(metrics[key]) for metrics in fold_metrics])
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }

        return summary

    def run_single_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Runs the automata pipeline for a single dataset."""
        self.logger.info(f"\nRUNNING PIPELINE FOR: {dataset_name.upper()}")

        data_dir = self.cfg.get("paths.data_dir")
        processed_dir = os.path.join(data_dir, "processed", dataset_name)

        if dataset_name == "skab":
            fold_metrics = []
            fold_dirs = sorted([
                item for item in os.listdir(processed_dir)
                if item.startswith("fold_")
            ])

            for fold_idx, fold_dir in enumerate(fold_dirs):
                fold_path = os.path.join(processed_dir, fold_dir)
                metrics = self._run_fold(dataset_name, fold_path, fold_idx)
                fold_metrics.append(metrics)

            summary_metrics = self._summarize_fold_metrics(fold_metrics)

            self.logger.info(f"--- SKAB (Avg over {len(fold_dirs)} folds) ---")
            self.logger.info(
                 "Accuracy: "
                f"{summary_metrics['accuracy']['mean']:.4f} +/- "
                f"{summary_metrics['accuracy']['std']:.4f}, "
                "F1: "
                 f"{summary_metrics['f1']['mean']:.4f} +/- "
               f"{summary_metrics['f1']['std']:.4f}"
)
        

            return {
                "dataset": dataset_name,
                "test_metrics": {
                    key: value["mean"]
                    for key, value in summary_metrics.items()
                },
                "test_metrics_std": {
                    key: value["std"]
                    for key, value in summary_metrics.items()
                },
                "fold_metrics": fold_metrics
            }

        metrics = self._run_fold(dataset_name, processed_dir, 0)

        self.logger.info("--- BATADAL ---")
        self.logger.info(
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics.get('precision', 0.0):.4f}, "
            f"Recall: {metrics.get('recall', 0.0):.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )

        return {
            "dataset": dataset_name,
            "test_metrics": metrics
        }

    def run_all(self) -> List[Dict[str, Any]]:
        """Runs the automata pipeline for all configured project datasets."""
        self.logger.start_timer("total_experiment_time")

        datasets = ["skab", "batadal"]
        results = []

        for dataset_name in datasets:
            try:
                result = self.run_single_dataset(dataset_name)
                results.append(result)
            except Exception as exc:
                self.logger.error(
                    f"Error running pipeline for {dataset_name}: {str(exc)}"
                )

        self.logger.stop_timer("total_experiment_time")
        return results