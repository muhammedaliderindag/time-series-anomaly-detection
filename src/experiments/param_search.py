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

class ParameterSearchTester:
    """
    Runs a grid search for Automata hyperparameters (window_size, alphabet_size)
    and logs the performance metrics and unique state counts.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)
        self.log_dir = self.cfg.get("paths.log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
    def run_grid_search(self) -> None:
        """
        Runs a grid search over window_size and alphabet_size for the Automata model.
        """
        print("--- Starting Automata Parameter Variation Testing ---")
        window_sizes = [3, 4, 5, 6]
        alphabet_sizes = [3, 4, 5, 6]
        
        paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)
        data_dir = self.cfg.get("paths.data_dir")
        
        datasets = ["swat", "wadi", "batadal"]
        all_results = []
        
        for ds in datasets:
            print(f"\n[Grid Search on Dataset]: {ds.upper()}")
            processed_dir = os.path.join(data_dir, "processed", ds)
            if not os.path.exists(processed_dir):
                print(f"Processed data for {ds} not found. Skipping...")
                continue
                
            train_pca = np.load(os.path.join(processed_dir, "train_pca.npy")).flatten()
            test_pca = np.load(os.path.join(processed_dir, "test_pca.npy")).flatten()
            test_lbl_orig = pd.read_csv(os.path.join(processed_dir, "test_labels.csv"))["label"].values
            
            for w in window_sizes:
                for a in alphabet_sizes:
                    print(f"  Testing Window={w}, Alphabet={a}...")
                    
                    # Transform
                    sax = SAXTransformer(segment_size=paa_segment_size, alphabet_size=a)
                    extractor = PatternExtractor(window_size=w)
                    
                    train_patterns = extractor.extract_patterns(sax.transform(train_pca))
                    test_patterns = extractor.extract_patterns(sax.transform(test_pca))
                    
                    # Train (we don't need to save to disk during grid search)
                    model = ProbabilisticAutomaton(model_dir=None)
                    model.fit(train_patterns)
                    
                    num_states = len(model.transition_matrix)
                    
                    # Evaluate
                    explainability = AutomataExplainability(model, anomaly_threshold)
                    test_justifications, _, _ = explainability.explain_path(test_patterns)
                    
                    test_labels = map_labels_to_patterns(test_lbl_orig, paa_segment_size, w)
                    test_preds = np.array([1 if j["decision"] == "anomaly" else 0 for j in test_justifications])
                    
                    metrics = calculate_metrics(test_labels, test_preds)
                    
                    res_dict = {
                        "dataset": ds,
                        "window_size": w,
                        "alphabet_size": a,
                        "unique_states": num_states,
                        "f1_score": metrics["f1"],
                        "accuracy": metrics["accuracy"]
                    }
                    all_results.append(res_dict)
                    
        # Log results
        log_path_json = os.path.join(self.log_dir, "automata_param_search.json")
        log_path_csv = os.path.join(self.log_dir, "automata_param_search.csv")
        
        with open(log_path_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)
            
        df = pd.DataFrame(all_results)
        df.to_csv(log_path_csv, index=False)
            
        print("\n--- Parameter Search Results ---")
        print(df.to_string(index=False))
        print(f"\nResults logged to {log_path_csv}")
