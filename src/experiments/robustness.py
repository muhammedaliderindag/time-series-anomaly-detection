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
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)
        self.log_dir = self.cfg.get("paths.log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def inject_gaussian_noise(self, data: np.ndarray, mean: float = 0.0, std: float = 0.1) -> np.ndarray:
        noise = np.random.normal(mean, std, size=data.shape)
        return data + noise

    def evaluate_automata(self, dataset_name: str, test_pca: np.ndarray, test_lbl_orig: np.ndarray, fold_idx: int = 0) -> Dict[str, float]:
        paa_segment_size = self.cfg.get("automata.paa_segment_size", 5)
        alphabet_size = self.cfg.get("automata.alphabet_size", 4)
        window_size = self.cfg.get("automata.window_size", 3)
        anomaly_threshold = self.cfg.get("automata.anomaly_threshold", 0.05)
        
        model_dir = self.cfg.get("paths.model_dir")
        model = ProbabilisticAutomaton(model_dir=model_dir)
        
        try:
            model_name = f"{dataset_name}_fold{fold_idx}_automaton.json" if dataset_name == "skab" else f"{dataset_name}_automaton.json"
            model.load(os.path.join(model.artifact_dir, model_name))
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

    def evaluate_dl(self, dataset_name: str, test_pca: np.ndarray, test_lbl_orig: np.ndarray, fold_idx: int = 0) -> Dict[str, float]:
        return {"f1": 0.0}

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

    def run_robustness_test(self, std_list: list = [0.1, 0.5]) -> None:
        print("--- Starting Robustness Testing (Gaussian Noise) ---")
        datasets = ["skab", "batadal"]
        results = []
        data_dir = self.cfg.get("paths.data_dir")
        
        for ds in datasets:
            print(f"Testing robustness for dataset: {ds}")
            test_pcas, test_lbls = self._get_test_data_for_ds(ds, data_dir)
            if not test_pcas: continue
            
            auto_orig_f1s = []
            dl_orig_f1s = []
            for i, (test_pca, test_lbl_orig) in enumerate(zip(test_pcas, test_lbls)):
                auto_orig_f1s.append(self.evaluate_automata(ds, test_pca, test_lbl_orig, i)["f1"])
                dl_orig_f1s.append(self.evaluate_dl(ds, test_pca, test_lbl_orig, i)["f1"])
            
            res_dict = {
                "dataset": ds,
                "automata_original_f1": np.mean(auto_orig_f1s),
                "dl_original_f1": np.mean(dl_orig_f1s)
            }
            
            for std in std_list:
                auto_noisy_f1s = []
                dl_noisy_f1s = []
                for i, (test_pca, test_lbl_orig) in enumerate(zip(test_pcas, test_lbls)):
                    noisy_test_pca = self.inject_gaussian_noise(test_pca, std=std)
                    auto_noisy_f1s.append(self.evaluate_automata(ds, noisy_test_pca, test_lbl_orig, i)["f1"])
                    dl_noisy_f1s.append(self.evaluate_dl(ds, noisy_test_pca, test_lbl_orig, i)["f1"])
                
                res_dict[f"automata_noise_{std}_f1"] = np.mean(auto_noisy_f1s)
                res_dict[f"dl_noise_{std}_f1"] = np.mean(dl_noisy_f1s)
                
            results.append(res_dict)
            
        log_path_json = os.path.join(self.log_dir, "robustness_test_results.json")
        log_path_csv = os.path.join(self.log_dir, "robustness_test_results.csv")
        with open(log_path_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        pd.DataFrame(results).to_csv(log_path_csv, index=False)
        print("\n--- Robustness Results ---")
        print(pd.DataFrame(results).to_string(index=False))
        print(f"\nResults logged to {log_path_json}\n")
