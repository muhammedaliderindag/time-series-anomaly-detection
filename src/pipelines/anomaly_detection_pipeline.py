"""
Anomaly Detection Pipeline Orchestrator
=======================================
Orchestrates the model training, validation, testing, evaluation,
and explainability outputs for the Time Series Anomaly Detection.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from src.utils.config_parser import ConfigParser
from src.utils.logger import ExperimentLogger
from src.utils.metrics import map_labels_to_patterns, calculate_metrics
from src.data.build_features import run_pipeline
from src.models.automata.transforms import SAXTransformer
from src.models.automata.pattern_extractor import PatternExtractor
from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton
from src.models.automata.explainability import AutomataExplainability


class AnomalyDetectionPipeline:
    """Manages z-normalized model training and inference pipelines across datasets."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.cfg = ConfigParser(config_path)
        self.logger = ExperimentLogger(self.cfg.config, experiment_name="automata_experiment")
        self.anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)
        self.paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        self.alphabet_size = self.cfg.get("automata.alphabet_size", 4)
        self.window_size = self.cfg.get("automata.window_size", 3)

    def run_preprocessing(self) -> None:
        """Runs the feature engineering/preprocessing step."""
        self.logger.info("Starting preprocessing step...")
        run_pipeline(self.config_path)  # Runs the build_features pipeline using the config path string
        self.logger.info("Preprocessing complete.")

    def run_single_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Runs train/val/test pipeline for a single dataset."""
        self.logger.info(f"\n========================================\n"
                         f"RUNNING PIPELINE FOR: {dataset_name.upper()}\n"
                         f"========================================")

        # 1. Load Data
        data_dir = self.cfg.get("paths.data_dir")
        processed_dir = os.path.join(data_dir, "processed", dataset_name)
        
        train_pca = np.load(os.path.join(processed_dir, "train_pca.npy")).flatten()
        val_pca = np.load(os.path.join(processed_dir, "val_pca.npy")).flatten()
        test_pca = np.load(os.path.join(processed_dir, "test_pca.npy")).flatten()
        
        train_lbl_orig = pd.read_csv(os.path.join(processed_dir, "train_labels.csv"))["label"].values
        val_lbl_orig = pd.read_csv(os.path.join(processed_dir, "val_labels.csv"))["label"].values
        test_lbl_orig = pd.read_csv(os.path.join(processed_dir, "test_labels.csv"))["label"].values

        # 2. Transform (SAX & Pattern Extraction)
        self.logger.info(f"Transforming z-score to SAX (segment_size={self.paa_segment_size}, "
                         f"alphabet={self.alphabet_size}, window={self.window_size})")

        sax = SAXTransformer(segment_size=self.paa_segment_size, alphabet_size=self.alphabet_size)
        extractor = PatternExtractor(window_size=self.window_size)

        train_patterns = extractor.extract_patterns(sax.transform(train_pca))
        val_patterns = extractor.extract_patterns(sax.transform(val_pca))
        test_patterns = extractor.extract_patterns(sax.transform(test_pca))

        self.logger.info(f"Patterns: Train={len(train_patterns)}, Val={len(val_patterns)}, Test={len(test_patterns)}")

        # 3. Fit Model
        model = ProbabilisticAutomaton(model_dir=self.cfg.get("paths.model_dir"))
        
        self.logger.start_timer(f"{dataset_name}_train_time")
        model.fit(train_patterns)
        self.logger.stop_timer(f"{dataset_name}_train_time")
        
        model_path = model.save(f"{dataset_name}_automaton.json")
        self.logger.info(f"Model saved to {model_path}")

        # 4. Inference & Explainability
        explainability = AutomataExplainability(model, self.anomaly_threshold)
        
        # Validation
        self.logger.info("Running validation inference...")
        val_justifications, _, _ = explainability.explain_path(val_patterns)
        val_exp_path = os.path.join(self.cfg.get("paths.log_dir"), f"{dataset_name}_val_explainability.json")
        with open(val_exp_path, "w", encoding="utf-8") as f:
            f.write(explainability.format_json_output(val_justifications))

        # Testing
        self.logger.info("Running test inference...")
        self.logger.start_timer(f"{dataset_name}_inference_time")
        test_justifications, _, _ = explainability.explain_path(test_patterns)
        self.logger.stop_timer(f"{dataset_name}_inference_time")
        
        test_exp_path = os.path.join(self.cfg.get("paths.log_dir"), f"{dataset_name}_test_explainability.json")
        with open(test_exp_path, "w", encoding="utf-8") as f:
            f.write(explainability.format_json_output(test_justifications))

        # 5. Evaluate Metrics
        val_labels = map_labels_to_patterns(val_lbl_orig, self.paa_segment_size, self.window_size)
        test_labels = map_labels_to_patterns(test_lbl_orig, self.paa_segment_size, self.window_size)

        val_preds = np.array([1 if j["decision"] == "anomaly" else 0 for j in val_justifications])
        test_preds = np.array([1 if j["decision"] == "anomaly" else 0 for j in test_justifications])

        val_metrics = calculate_metrics(val_labels, val_preds)
        test_metrics = calculate_metrics(test_labels, test_preds)

        # Log to logger
        self.logger.log_metrics(
            seed=self.cfg.get("random_seeds")[0],
            fold=0,
            val_accuracy=val_metrics["accuracy"],
            val_f1=val_metrics["f1"],
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
            test_accuracy=test_metrics["accuracy"],
            test_f1=test_metrics["f1"],
            test_precision=test_metrics["precision"],
            test_recall=test_metrics["recall"],
        )

        self.logger.info(f"--- TEST RESULTS ({dataset_name.upper()}) ---")
        self.logger.info(f"Accuracy:  {test_metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {test_metrics['precision']:.4f}")
        self.logger.info(f"Recall:    {test_metrics['recall']:.4f}")
        self.logger.info(f"F1-score:  {test_metrics['f1']:.4f}")

        return {
            "dataset": dataset_name,
            "test_metrics": test_metrics
        }

    def run_all(self) -> List[Dict[str, Any]]:
        """Executes pipelines across all datasets (SWAT, WADI, BATADAL)."""
        self.logger.start_timer("total_experiment_time")
        
        datasets = ["swat", "wadi", "batadal"]
        results = []
        
        for ds in datasets:
            try:
                res = self.run_single_dataset(ds)
                results.append(res)
            except Exception as e:
                self.logger.error(f"Error running pipeline for {ds}: {str(e)}")

        self.logger.stop_timer("total_experiment_time")
        self.logger.save_metrics(fmt="both")
        self.logger.summary()
        
        return results
