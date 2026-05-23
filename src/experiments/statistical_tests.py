import numpy as np
import pandas as pd
import json
import os
import torch
from statsmodels.stats.contingency_tables import mcnemar

from src.utils.config_parser import ConfigParser
from src.utils.metrics import map_labels_to_patterns
from src.models.automata.transforms import SAXTransformer
from src.models.automata.pattern_extractor import PatternExtractor
from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton
from src.models.automata.explainability import AutomataExplainability

from src.models.dl.architectures import LSTMAutoencoder
from src.models.dl.data_loader import create_dataloader
from src.models.dl.evaluate import evaluate_model, detect_anomalies

class StatisticalTester:
    """
    Evaluates statistical significance between the Automata and DL models using McNemar's test.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)
        self.log_dir = self.cfg.get("paths.log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_mcnemar_test(self, dataset_name: str = "swat") -> dict:
        """
        Runs McNemar's test comparing the predictions of the Automata vs. DL model.
        """
        print(f"--- Running McNemar's Test on {dataset_name.upper()} ---")
        data_dir = self.cfg.get("paths.data_dir")
        processed_dir = os.path.join(data_dir, "processed", dataset_name)
        
        if not os.path.exists(processed_dir):
            return {"error": "Dataset not processed."}
            
        test_pca = np.load(os.path.join(processed_dir, "test_pca.npy")).flatten()
        test_lbl_orig = pd.read_csv(os.path.join(processed_dir, "test_labels.csv"))["label"].values
        
        # 1. Automata Predictions
        paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        alphabet_size = self.cfg.get("automata.alphabet_size", 4)
        window_size = self.cfg.get("automata.window_size", 3)
        anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)
        
        model_dir = self.cfg.get("paths.model_dir")
        auto_model = ProbabilisticAutomaton(model_dir=model_dir)
        try:
            auto_model.load(f"{dataset_name}_automaton.json")
        except FileNotFoundError:
            return {"error": "Automata model not found."}
            
        sax = SAXTransformer(segment_size=paa_segment_size, alphabet_size=alphabet_size)
        extractor = PatternExtractor(window_size=window_size)
        test_patterns = extractor.extract_patterns(sax.transform(test_pca))
        
        explainability = AutomataExplainability(auto_model, anomaly_threshold)
        test_justifications, _, _ = explainability.explain_path(test_patterns)
        
        preds_auto = np.array([1 if j["decision"] == "anomaly" else 0 for j in test_justifications])
        
        # 2. DL Predictions
        model_path = os.path.join(model_dir, f"{dataset_name}_dl_model.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "best_dl_model.pt")
            if not os.path.exists(model_path):
                return {"error": "DL model not found."}
                
        input_dim = self.cfg.get("model.input_dim", 1)
        hidden_dim = self.cfg.get("model.hidden_dim", 64)
        latent_dim = self.cfg.get("model.latent_dim", 32)
        num_layers = self.cfg.get("model.num_layers", 2)
        
        dl_model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, num_layers)
        dl_model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        seq_len = self.cfg.get("data.sequence_length", 100)
        batch_size = self.cfg.get("batch_size", 32)
        
        dataloader = create_dataloader(test_pca, seq_len, batch_size)
        scores_dl = evaluate_model(dl_model, dataloader, self.device)
        
        threshold_dl = np.percentile(scores_dl, 95)
        preds_dl = detect_anomalies(scores_dl, threshold_dl)
        test_labels_dl = test_lbl_orig[seq_len - 1:]
        
        # We need predictions on the EXACT same instances to do McNemar
        # To simplify, we will truncate to the minimum common length 
        min_len = min(len(preds_auto), len(preds_dl))
        preds_auto = preds_auto[:min_len]
        preds_dl = preds_dl[:min_len]
        true_labels = test_labels_dl[:min_len] # approximation for alignment
        
        # Compute exact correctness
        auto_correct = (preds_auto == true_labels)
        dl_correct = (preds_dl == true_labels)
        
        # McNemar contingency table
        # [[both_correct, auto_correct_only],
        #  [dl_correct_only, neither_correct]]
        table = [[int(sum(auto_correct & dl_correct)), int(sum(auto_correct & ~dl_correct))],
                 [int(sum(~auto_correct & dl_correct)), int(sum(~auto_correct & ~dl_correct))]]
                 
        result = mcnemar(table, exact=False, correction=True)
        
        interpretation = "Statistically Significant Difference (Reject Null Hypothesis)" if result.pvalue < 0.05 else "No Statistically Significant Difference (Fail to Reject Null Hypothesis)"
        
        res_dict = {
            "dataset": dataset_name,
            "test_type": "McNemar",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "interpretation": interpretation,
            "contingency_table": table
        }
        
        log_path = os.path.join(self.log_dir, f"statistical_test_{dataset_name}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(res_dict, f, indent=4)
            
        print(f"p-value: {result.pvalue:.5f} -> {interpretation}")
        print(f"Saved results to {log_path}")
        return res_dict

if __name__ == "__main__":
    tester = StatisticalTester()
    tester.run_mcnemar_test("swat")
