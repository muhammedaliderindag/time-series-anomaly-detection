import numpy as np
import pandas as pd
import json
import os
import torch
from typing import Dict, Any

from src.utils.config_parser import ConfigParser
from src.utils.metrics import calculate_metrics, map_labels_to_patterns
from src.models.automata.transforms import SAXTransformer
from src.models.automata.pattern_extractor import PatternExtractor
from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton
from src.models.automata.explainability import AutomataExplainability

from src.models.dl.architectures import LSTMAutoencoder
from src.models.dl.data_loader import create_dataloader
from src.models.dl.evaluate import evaluate_model, detect_anomalies

class RobustnessTester:
    """
    Evaluates model robustness against Gaussian noise injected into the test data.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)
        self.log_dir = self.cfg.get("paths.log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def inject_gaussian_noise(self, data: np.ndarray, mean: float = 0.0, std: float = 0.1) -> np.ndarray:
        """Injects Gaussian noise into the given dataset."""
        noise = np.random.normal(mean, std, size=data.shape)
        return data + noise

    def evaluate_automata(self, dataset_name: str, test_pca: np.ndarray, test_lbl_orig: np.ndarray) -> Dict[str, float]:
        """Evaluates the pre-trained Automata model on the given test set."""
        paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        alphabet_size = self.cfg.get("automata.alphabet_size", 4)
        window_size = self.cfg.get("automata.window_size", 3)
        anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)
        
        model_dir = self.cfg.get("paths.model_dir")
        model = ProbabilisticAutomaton(model_dir=model_dir)
        
        try:
            model.load(f"{dataset_name}_automaton.json")
        except FileNotFoundError:
            return {"f1": 0.0}
            
        sax = SAXTransformer(segment_size=paa_segment_size, alphabet_size=alphabet_size)
        extractor = PatternExtractor(window_size=window_size)
        test_patterns = extractor.extract_patterns(sax.transform(test_pca))
        
        explainability = AutomataExplainability(model, anomaly_threshold)
        test_justifications, _, _ = explainability.explain_path(test_patterns)
        
        test_labels = map_labels_to_patterns(test_lbl_orig, paa_segment_size, window_size)
        test_preds = np.array([1 if j["decision"] == "anomaly" else 0 for j in test_justifications])
        
        return calculate_metrics(test_labels, test_preds)

    def evaluate_dl(self, dataset_name: str, test_pca: np.ndarray, test_lbl_orig: np.ndarray) -> Dict[str, float]:
        """Evaluates the DL model on the given test set."""
        model_dir = self.cfg.get("paths.model_dir", "./models")
        # DL models usually saved with dataset prefix. 
        model_path = os.path.join(model_dir, f"{dataset_name}_dl_model.pt")
        
        if not os.path.exists(model_path):
            # Fallback to the generic one created by trainer
            model_path = os.path.join(model_dir, "best_dl_model.pt")
            if not os.path.exists(model_path):
                return {"f1": 0.0}

        # Initialize model
        input_dim = self.cfg.get("model.input_dim", 1)
        hidden_dim = self.cfg.get("model.hidden_dim", 64)
        latent_dim = self.cfg.get("model.latent_dim", 32)
        num_layers = self.cfg.get("model.num_layers", 2)
        
        model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, num_layers)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        seq_len = self.cfg.get("data.sequence_length", 100)
        batch_size = self.cfg.get("batch_size", 32)
        
        dataloader = create_dataloader(test_pca, seq_len, batch_size)
        
        scores = evaluate_model(model, dataloader, self.device)
        
        # We need a threshold. Since we don't have train_scores here, use a heuristic 95th percentile
        threshold = np.percentile(scores, 95) 
        preds = detect_anomalies(scores, threshold)
        
        # Adjust labels to match windowing: drop first `seq_len - 1`
        test_lbl_adjusted = test_lbl_orig[seq_len - 1:]
        
        # Saftey check for length mismatch (can happen with windowing)
        min_len = min(len(test_lbl_adjusted), len(preds))
        
        return calculate_metrics(test_lbl_adjusted[:min_len], preds[:min_len])

    def run_robustness_test(self, std_list: list = [0.1, 0.5]) -> None:
        """
        Evaluates models on original vs noisy test datasets.
        """
        print("--- Starting Robustness Testing (Gaussian Noise) ---")
        datasets = ["swat", "wadi", "batadal"]
        results = []
        
        data_dir = self.cfg.get("paths.data_dir")
        
        for ds in datasets:
            print(f"Testing robustness for dataset: {ds}")
            processed_dir = os.path.join(data_dir, "processed", ds)
            if not os.path.exists(processed_dir):
                print(f"Processed data for {ds} not found. Skipping...")
                continue
                
            test_pca = np.load(os.path.join(processed_dir, "test_pca.npy")).flatten()
            test_lbl_orig = pd.read_csv(os.path.join(processed_dir, "test_labels.csv"))["label"].values
            
            # Automata Eval
            orig_metrics_automata = self.evaluate_automata(ds, test_pca, test_lbl_orig)
            orig_metrics_dl = self.evaluate_dl(ds, test_pca, test_lbl_orig)
            
            res_dict = {
                "dataset": ds,
                "automata_original_f1": orig_metrics_automata["f1"],
                "dl_original_f1": orig_metrics_dl["f1"]
            }
            
            # Noisy Eval
            for std in std_list:
                noisy_test_pca = self.inject_gaussian_noise(test_pca, std=std)
                
                noisy_metrics_auto = self.evaluate_automata(ds, noisy_test_pca, test_lbl_orig)
                noisy_metrics_dl = self.evaluate_dl(ds, noisy_test_pca, test_lbl_orig)
                
                res_dict[f"automata_noise_{std}_f1"] = noisy_metrics_auto["f1"]
                res_dict[f"dl_noise_{std}_f1"] = noisy_metrics_dl["f1"]
                
            results.append(res_dict)
            
        # Log results
        log_path_json = os.path.join(self.log_dir, "robustness_test_results.json")
        log_path_csv = os.path.join(self.log_dir, "robustness_test_results.csv")
        
        with open(log_path_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        df = pd.DataFrame(results)
        df.to_csv(log_path_csv, index=False)
            
        print("\n--- Robustness Results ---")
        print(df.to_string(index=False))
        print(f"\nResults logged to {log_path_json}\n")
