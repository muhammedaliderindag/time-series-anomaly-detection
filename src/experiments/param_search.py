import numpy as np
import pandas as pd
import json
import os

from src.utils.config_parser import ConfigParser
from src.utils.metrics import calculate_metrics, map_labels_to_patterns
from src.models.automata.transforms import SAXTransformer
from src.models.automata.pattern_extractor import PatternExtractor
from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton
from src.models.automata.explainability import AutomataExplainability

class ParameterSearchTester:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)
        self.log_dir = self.cfg.get("paths.log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
    def _evaluate_dir(self, ds_dir: str, w: int, a: int, paa_segment_size: int, anomaly_threshold: float):
        train_pca = np.load(os.path.join(ds_dir, "train_pca.npy")).flatten()
        test_pca = np.load(os.path.join(ds_dir, "test_pca.npy")).flatten()
        test_lbl = pd.read_csv(os.path.join(ds_dir, "test_labels.csv"))["label"].values if os.path.exists(os.path.join(ds_dir, "test_labels.csv")) else np.zeros(len(test_pca))
        
        sax = SAXTransformer(segment_size=paa_segment_size, alphabet_size=a)
        extractor = PatternExtractor(window_size=w)
        
        train_patterns = extractor.extract_patterns(sax.transform(train_pca))
        test_patterns = extractor.extract_patterns(sax.transform(test_pca))
        
        model = ProbabilisticAutomaton()
        model.fit(train_patterns)
        num_states = len(model.transition_matrix)
        
        explainability = AutomataExplainability(model, anomaly_threshold)
        test_justifications, _, _ = explainability.explain_path(test_patterns)
        
        test_labels = map_labels_to_patterns(test_lbl, paa_segment_size, w)
        test_preds = np.array([1 if j["decision"] == "anomaly" else 0 for j in test_justifications])
        
        metrics = calculate_metrics(test_labels, test_preds)
        return num_states, metrics

    def run_grid_search(self) -> None:
        print("--- Starting Automata Parameter Variation Testing ---")
        window_sizes = [3, 4, 5, 6]
        alphabet_sizes = [3, 4, 5, 6]
        paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)
        data_dir = self.cfg.get("paths.data_dir")
        
        datasets = ["skab", "batadal"]
        all_results = []
        
        for ds in datasets:
            print(f"\n[Grid Search on Dataset]: {ds.upper()}")
            processed_dir = os.path.join(data_dir, "processed", ds)
            
            for w in window_sizes:
                for a in alphabet_sizes:
                    print(f"  Testing Window={w}, Alphabet={a}...")
                    if ds == "skab":
                        fold_metrics = []
                        fold_states = []
                        folds_dirs = [d for d in os.listdir(processed_dir) if d.startswith("fold_")]
                        for fdir in folds_dirs:
                            states, metrics = self._evaluate_dir(os.path.join(processed_dir, fdir), w, a, paa_segment_size, anomaly_threshold)
                            fold_states.append(states)
                            fold_metrics.append(metrics)
                        
                        avg_states = np.mean(fold_states)
                        avg_f1 = np.mean([m["f1"] for m in fold_metrics])
                        avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
                        
                        all_results.append({"dataset": ds, "window_size": w, "alphabet_size": a, "unique_states": avg_states, "f1_score": avg_f1, "accuracy": avg_acc})
                    else:
                        states, metrics = self._evaluate_dir(processed_dir, w, a, paa_segment_size, anomaly_threshold)
                        all_results.append({"dataset": ds, "window_size": w, "alphabet_size": a, "unique_states": states, "f1_score": metrics["f1"], "accuracy": metrics["accuracy"]})
                    
        log_path_json = os.path.join(self.log_dir, "automata_param_search.json")
        log_path_csv = os.path.join(self.log_dir, "automata_param_search.csv")
        with open(log_path_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)
        pd.DataFrame(all_results).to_csv(log_path_csv, index=False)
        print(f"\nResults logged to {log_path_csv}")
