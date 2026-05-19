import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Any

from src.utils.config_parser import ConfigParser
from src.utils.metrics import calculate_metrics, map_labels_to_patterns
from src.models.automata.transforms import SAXTransformer
from src.models.automata.pattern_extractor import PatternExtractor
from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton
from src.models.automata.explainability import AutomataExplainability

class CrossDatasetTester:
    """
    Tests model generalization by evaluating a model trained on one dataset
    against the test splits of other datasets.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)
        self.log_dir = self.cfg.get("paths.log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.datasets = ["swat", "wadi", "batadal"]
        
    def evaluate_cross_dataset(self) -> None:
        """
        Trains/loads a model on one dataset and evaluates it on the test splits of the others.
        """
        print("--- Starting Cross-Dataset Generalization Testing ---")
        
        paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        alphabet_size = self.cfg.get("automata.alphabet_size", 4)
        window_size = self.cfg.get("automata.window_size", 3)
        anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)
        
        model_dir = self.cfg.get("paths.model_dir")
        data_dir = self.cfg.get("paths.data_dir")
        
        # Matrix to hold results: rows = Train, cols = Test
        results_matrix = []
        
        for train_ds in self.datasets:
            print(f"\n[Source Model]: {train_ds.upper()}")
            
            # Load the source model
            model = ProbabilisticAutomaton(model_dir=model_dir)
            try:
                model.load(f"{train_ds}_automaton.json")
            except FileNotFoundError:
                print(f"Model for {train_ds} not found. Skipping as source.")
                continue
                
            row_results = {"train_dataset": train_ds}
            
            for test_ds in self.datasets:
                processed_dir = os.path.join(data_dir, "processed", test_ds)
                if not os.path.exists(processed_dir):
                    row_results[test_ds] = None
                    continue
                    
                test_pca = np.load(os.path.join(processed_dir, "test_pca.npy")).flatten()
                test_lbl_orig = pd.read_csv(os.path.join(processed_dir, "test_labels.csv"))["label"].values
                
                sax = SAXTransformer(segment_size=paa_segment_size, alphabet_size=alphabet_size)
                extractor = PatternExtractor(window_size=window_size)
                test_patterns = extractor.extract_patterns(sax.transform(test_pca))
                
                explainability = AutomataExplainability(model, anomaly_threshold)
                test_justifications, _, _ = explainability.explain_path(test_patterns)
                
                test_labels = map_labels_to_patterns(test_lbl_orig, paa_segment_size, window_size)
                test_preds = np.array([1 if j["decision"] == "anomaly" else 0 for j in test_justifications])
                
                metrics = calculate_metrics(test_labels, test_preds)
                f1_score = metrics["f1"]
                
                row_results[test_ds] = f1_score
                print(f"  -> Evaluated on {test_ds.upper()}: F1 = {f1_score:.4f}")
                
            results_matrix.append(row_results)
            
        # Log results
        log_path_json = os.path.join(self.log_dir, "cross_dataset_results.json")
        log_path_csv = os.path.join(self.log_dir, "cross_dataset_matrix.csv")
        
        with open(log_path_json, "w", encoding="utf-8") as f:
            json.dump(results_matrix, f, indent=4)
            
        df = pd.DataFrame(results_matrix)
        df.to_csv(log_path_csv, index=False)
            
        print("\n--- Cross-Dataset Matrix (F1-Scores) ---")
        print(df.to_string(index=False))
        print(f"\nResults logged to {log_path_csv}")
