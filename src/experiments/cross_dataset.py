import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Any, List

from src.utils.config_parser import ConfigParser
from src.utils.metrics import calculate_metrics, map_labels_to_patterns
from src.models.automata.transforms import SAXTransformer
from src.models.automata.pattern_extractor import PatternExtractor
from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton
from src.models.automata.explainability import AutomataExplainability

class CrossDatasetTester:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)
        self.log_dir = self.cfg.get("paths.log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.datasets = ["skab", "batadal"]
        
    def _get_test_data_for_ds(self, ds: str, data_dir: str):
        processed_dir = os.path.join(data_dir, "processed", ds)
        if ds == "skab":
            test_pcas = []
            test_lbls = []
            for fdir in [d for d in os.listdir(processed_dir) if d.startswith("fold_")]:
                fpath = os.path.join(processed_dir, fdir)
                test_pcas.append(np.load(os.path.join(fpath, "test_pca.npy")).flatten())
                if os.path.exists(os.path.join(fpath, "test_labels.csv")):
                    test_lbls.append(pd.read_csv(os.path.join(fpath, "test_labels.csv"))["label"].values)
                else:
                    test_lbls.append(np.zeros(len(test_pcas[-1])))
            return test_pcas, test_lbls
        else:
            if not os.path.exists(os.path.join(processed_dir, "test_pca.npy")):
                return [], []
            test_pca = [np.load(os.path.join(processed_dir, "test_pca.npy")).flatten()]
            if os.path.exists(os.path.join(processed_dir, "test_labels.csv")):
                test_lbl = [pd.read_csv(os.path.join(processed_dir, "test_labels.csv"))["label"].values]
            else:
                test_lbl = [np.zeros(len(test_pca[0]))]
            return test_pca, test_lbl

    def evaluate_cross_dataset(self) -> None:
        print("--- Starting Cross-Dataset Generalization Testing ---")
        
        paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        alphabet_size = self.cfg.get("automata.alphabet_size", 4)
        window_size = self.cfg.get("automata.window_size", 3)
        anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)
        
        model_dir = self.cfg.get("paths.model_dir")
        data_dir = self.cfg.get("paths.data_dir")
        
        results_matrix = []
        
        for train_ds in self.datasets:
            print(f"\n[Source Model]: {train_ds.upper()}")
            model = ProbabilisticAutomaton(model_dir=model_dir)
            try:
                model_name = f"{train_ds}_fold0_automaton.json" if train_ds == "skab" else f"{train_ds}_automaton.json"
                model_path = os.path.join(model.artifact_dir, model_name)
                if not os.path.exists(model_path):
                     model_name = f"{train_ds}_fold0_automaton.json"
                     model_path = os.path.join(model.artifact_dir, model_name)
                model.load(model_path)
            except FileNotFoundError:
                print(f"Model for {train_ds} not found. Skipping as source.")
                continue
                
            row_results = {"train_dataset": train_ds}
            
            for test_ds in self.datasets:
                test_pcas, test_lbls = self._get_test_data_for_ds(test_ds, data_dir)
                if not test_pcas:
                    row_results[test_ds] = None
                    continue
                
                f1_scores = []
                for test_pca, test_lbl_orig in zip(test_pcas, test_lbls):
                    sax = SAXTransformer(segment_size=paa_segment_size, alphabet_size=alphabet_size)
                    extractor = PatternExtractor(window_size=window_size)
                    test_patterns = extractor.extract_patterns(sax.transform(test_pca))
                    
                    explainability = AutomataExplainability(model, anomaly_threshold)
                    test_justifications, _, _ = explainability.explain_path(test_patterns)
                    
                    test_labels = map_labels_to_patterns(test_lbl_orig, paa_segment_size, window_size)
                    test_preds = np.array([1 if j["decision"] == "anomaly" else 0 for j in test_justifications])
                    
                    metrics = calculate_metrics(test_labels, test_preds)
                    f1_scores.append(metrics["f1"])
                
                avg_f1 = np.mean(f1_scores)
                row_results[test_ds] = avg_f1
                print(f"  -> Evaluated on {test_ds.upper()}: F1 = {avg_f1:.4f}")
                
            results_matrix.append(row_results)
            
        log_path_json = os.path.join(self.log_dir, "cross_dataset_results.json")
        log_path_csv = os.path.join(self.log_dir, "cross_dataset_matrix.csv")
        
        with open(log_path_json, "w", encoding="utf-8") as f:
            json.dump(results_matrix, f, indent=4)
            
        pd.DataFrame(results_matrix).to_csv(log_path_csv, index=False)
        print("\n--- Cross-Dataset Matrix (F1-Scores) ---")
        print(pd.DataFrame(results_matrix).to_string(index=False))
        print(f"\nResults logged to {log_path_csv}")
