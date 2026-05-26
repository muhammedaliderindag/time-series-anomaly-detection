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
        self.config_path = config_path
        self.cfg = ConfigParser(config_path)
        self.logger = ExperimentLogger(self.cfg.config, experiment_name="automata_experiment")
        self.anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)
        self.paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        self.alphabet_size = self.cfg.get("automata.alphabet_size", 4)
        self.window_size = self.cfg.get("automata.window_size", 3)

    def run_preprocessing(self) -> None:
        self.logger.info("Starting preprocessing step...")
        run_pipeline(self.config_path)
        self.logger.info("Preprocessing complete.")

    def _run_fold(self, dataset_name: str, fold_dir: str, fold_idx: int) -> Dict[str, Any]:
        train_pca = np.load(os.path.join(fold_dir, "train_pca.npy")).flatten()
        val_pca = np.load(os.path.join(fold_dir, "val_pca.npy")).flatten()
        test_pca = np.load(os.path.join(fold_dir, "test_pca.npy")).flatten()
        
        train_lbl = pd.read_csv(os.path.join(fold_dir, "train_labels.csv"))["label"].values if os.path.exists(os.path.join(fold_dir, "train_labels.csv")) else np.zeros(len(train_pca))
        val_lbl = pd.read_csv(os.path.join(fold_dir, "val_labels.csv"))["label"].values if os.path.exists(os.path.join(fold_dir, "val_labels.csv")) else np.zeros(len(val_pca))
        test_lbl = pd.read_csv(os.path.join(fold_dir, "test_labels.csv"))["label"].values if os.path.exists(os.path.join(fold_dir, "test_labels.csv")) else np.zeros(len(test_pca))

        sax = SAXTransformer(segment_size=self.paa_segment_size, alphabet_size=self.alphabet_size)
        extractor = PatternExtractor(window_size=self.window_size)

        train_patterns = extractor.extract_patterns(sax.transform(train_pca))
        val_patterns = extractor.extract_patterns(sax.transform(val_pca))
        test_patterns = extractor.extract_patterns(sax.transform(test_pca))

        model = ProbabilisticAutomaton(model_dir=self.cfg.get("paths.model_dir"))
        model.fit(train_patterns)
        model.save(f"{dataset_name}_fold{fold_idx}_automaton.json")

        explainability = AutomataExplainability(model, self.anomaly_threshold)
        
        # Test Inference
        test_justifications, _, _ = explainability.explain_path(test_patterns)
        
        test_labels = map_labels_to_patterns(test_lbl, self.paa_segment_size, self.window_size)
        test_preds = np.array([1 if j["decision"] == "anomaly" else 0 for j in test_justifications])
        
        test_metrics = calculate_metrics(test_labels, test_preds)
        return test_metrics

    def run_single_dataset(self, dataset_name: str) -> Dict[str, Any]:
        self.logger.info(f"\nRUNNING PIPELINE FOR: {dataset_name.upper()}")
        data_dir = self.cfg.get("paths.data_dir")
        processed_dir = os.path.join(data_dir, "processed", dataset_name)
        
        if dataset_name == "skab":
            fold_metrics = []
            folds_dirs = [d for d in os.listdir(processed_dir) if d.startswith("fold_")]
            for i, fdir in enumerate(folds_dirs):
                fpath = os.path.join(processed_dir, fdir)
                metrics = self._run_fold(dataset_name, fpath, i)
                fold_metrics.append(metrics)
            
            # Average metrics
            avg_metrics = {}
            for k in fold_metrics[0].keys():
                avg_metrics[k] = np.mean([m[k] for m in fold_metrics])
            
            self.logger.info(f"--- SKAB (Avg over {len(folds_dirs)} folds) ---")
            self.logger.info(f"Accuracy: {avg_metrics['accuracy']:.4f}, F1: {avg_metrics['f1']:.4f}")
            return {"dataset": dataset_name, "test_metrics": avg_metrics}
        else:
            metrics = self._run_fold(dataset_name, processed_dir, 0)
            self.logger.info(f"--- BATADAL ---")
            self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            return {"dataset": dataset_name, "test_metrics": metrics}

    def run_all(self) -> List[Dict[str, Any]]:
        self.logger.start_timer("total_experiment_time")
        datasets = ["skab", "batadal"]
        results = []
        for ds in datasets:
            try:
                res = self.run_single_dataset(ds)
                results.append(res)
            except Exception as e:
                self.logger.error(f"Error running pipeline for {ds}: {str(e)}")

        self.logger.stop_timer("total_experiment_time")
        return results
