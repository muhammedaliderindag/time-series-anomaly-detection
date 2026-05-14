"""
Main Pipeline Entrypoint
=========================
Executes the end-to-end Machine Learning pipeline:
1. Data Preprocessing (loading, splitting, scaling, PCA, label extraction).
2. SAX discretization and pattern extraction.
3. Training Probabilistic Automaton models on the Train splits.
4. Validation and Testing (inference, Levenshtein unseen mapping, decision rule).
5. Explainability reporting (saving step-by-step justifications as JSON).
6. Performance metrics logging and summary reporting.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_parser import ConfigParser
from src.utils.logger import ExperimentLogger
from src.data.build_features import run_pipeline
from src.models.automata.transforms import SAXTransformer
from src.models.automata.pattern_extractor import PatternExtractor
from src.models.automata.probabilistic_automaton import ProbabilisticAutomaton
from src.models.automata.explainability import AutomataExplainability


def map_labels_to_patterns(original_labels: np.ndarray, segment_size: int, window_size: int) -> np.ndarray:
    """
    Downsamples the original ground truth labels to align with SAX pattern windows.
    A pattern window is labeled an anomaly if ANY original time-step in it is an anomaly.
    """
    n_original = len(original_labels)
    n_paa = n_original // segment_size
    n_patterns = n_paa - window_size + 1
    
    pattern_labels = []
    for i in range(n_patterns):
        start_idx = i * segment_size
        end_idx = (i + window_size) * segment_size
        is_anomaly = 1 if np.any(original_labels[start_idx:end_idx] == 1) else 0
        pattern_labels.append(is_anomaly)
        
    return np.array(pattern_labels)


def run_experiment(cfg: ConfigParser, logger: ExperimentLogger, dataset_name: str) -> dict:
    """Runs train/val/test evaluation on a single dataset."""
    logger.info(f"\n========================================\n"
                f"RUNNING EXPERIMENT FOR: {dataset_name.upper()}\n"
                f"========================================")

    # 1. Resolve Paths
    data_dir = cfg.get("paths.data_dir")
    processed_dir = os.path.join(data_dir, "processed", dataset_name)
    
    # Load 1D PCA features
    train_pca = np.load(os.path.join(processed_dir, "train_pca.npy")).flatten()
    val_pca = np.load(os.path.join(processed_dir, "val_pca.npy")).flatten()
    test_pca = np.load(os.path.join(processed_dir, "test_pca.npy")).flatten()
    
    # Load labels
    train_lbl_orig = pd.read_csv(os.path.join(processed_dir, "train_labels.csv"))["label"].values
    val_lbl_orig = pd.read_csv(os.path.join(processed_dir, "val_labels.csv"))["label"].values
    test_lbl_orig = pd.read_csv(os.path.join(processed_dir, "test_labels.csv"))["label"].values

    # 2. SAX Discretization & Pattern Extraction
    paa_segment_size = cfg.get("automata.paa_segment_size", 5)
    alphabet_size = cfg.get("automata.alphabet_size", 4)
    window_size = cfg.get("automata.window_size", 3)
    anomaly_threshold = cfg.get("automata.anomaly_threshold", 0.05)

    logger.info(f"Extracting SAX patterns (segment_size={paa_segment_size}, "
                f"alphabet={alphabet_size}, window={window_size})")

    sax_transformer = SAXTransformer(segment_size=paa_segment_size, alphabet_size=alphabet_size)
    pattern_extractor = PatternExtractor(window_size=window_size)

    # Transformed symbolic strings
    train_sax = sax_transformer.transform(train_pca)
    val_sax = sax_transformer.transform(val_pca)
    test_sax = sax_transformer.transform(test_pca)

    # Extracted pattern lists (states sequence)
    train_patterns = pattern_extractor.extract_patterns(train_sax)
    val_patterns = pattern_extractor.extract_patterns(val_sax)
    test_patterns = pattern_extractor.extract_patterns(test_sax)

    logger.info(f"Extracted patterns counts: Train={len(train_patterns)}, "
                f"Val={len(val_patterns)}, Test={len(test_patterns)}")

    # 3. Model Training (Fit on Train ONLY)
    logger.info("Training Probabilistic Automaton...")
    model = ProbabilisticAutomaton(model_dir=cfg.get("paths.model_dir"))
    
    logger.start_timer(f"{dataset_name}_train_time")
    model.fit(train_patterns)
    logger.stop_timer(f"{dataset_name}_train_time")

    # Save model
    model_path = model.save(f"{dataset_name}_automaton.json")
    logger.info(f"Saved automaton model to {model_path}")

    # 4. Explainability & Inference (Validation)
    logger.info("Running inference on Validation set...")
    explainability = AutomataExplainability(model, anomaly_threshold)
    
    val_justifications, val_path_prob, val_conf = explainability.explain_path(val_patterns)

    # Save validation explainability JSON
    val_exp_path = os.path.join(cfg.get("paths.log_dir"), f"{dataset_name}_val_explainability.json")
    with open(val_exp_path, "w", encoding="utf-8") as f:
        f.write(explainability.format_json_output(val_justifications))
    logger.info(f"Saved validation explainability justification to {val_exp_path}")

    # 5. Explainability & Inference (Testing)
    logger.info("Running inference on Test set...")
    logger.start_timer(f"{dataset_name}_inference_time")
    test_justifications, test_path_prob, test_conf = explainability.explain_path(test_patterns)
    logger.stop_timer(f"{dataset_name}_inference_time")

    # Save test explainability JSON
    test_exp_path = os.path.join(cfg.get("paths.log_dir"), f"{dataset_name}_test_explainability.json")
    with open(test_exp_path, "w", encoding="utf-8") as f:
        f.write(explainability.format_json_output(test_justifications))
    logger.info(f"Saved test explainability justification to {test_exp_path}")

    # 6. Align Ground Truth Labels to Pattern Sequences
    val_labels = map_labels_to_patterns(val_lbl_orig, paa_segment_size, window_size)
    test_labels = map_labels_to_patterns(test_lbl_orig, paa_segment_size, window_size)

    # 7. Evaluate Predictions
    val_preds = np.array([1 if j["decision"] == "anomaly" else 0 for j in val_justifications])
    test_preds = np.array([1 if j["decision"] == "anomaly" else 0 for j in test_justifications])

    # Handle edge case where first state has no transition (length match padding)
    if len(val_preds) < len(val_labels):
        val_labels = val_labels[:len(val_preds)]
    if len(test_preds) < len(test_labels):
        test_labels = test_labels[:len(test_preds)]

    # Compute validation metrics
    val_acc = accuracy_score(val_labels, val_preds)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary', zero_division=0)

    # Compute test metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary', zero_division=0)

    # Log metrics
    logger.log_metrics(
        seed=cfg.get("random_seeds")[0],
        fold=0,
        val_accuracy=float(val_acc),
        val_f1=float(val_f1),
        val_precision=float(val_prec),
        val_recall=float(val_rec),
        test_accuracy=float(test_acc),
        test_f1=float(test_f1),
        test_precision=float(test_prec),
        test_recall=float(test_rec),
    )

    logger.info(f"--- TEST SET RESULTS ({dataset_name.upper()}) ---")
    logger.info(f"Accuracy:  {test_acc:.4f}")
    logger.info(f"Precision: {test_prec:.4f}")
    logger.info(f"Recall:    {test_rec:.4f}")
    logger.info(f"F1-score:  {test_f1:.4f}")

    return {
        "dataset": dataset_name,
        "test_acc": test_acc,
        "test_prec": test_prec,
        "test_rec": test_rec,
        "test_f1": test_f1
    }


def main():
    print("=" * 60)
    print("STARTING TIME SERIES ANOMALY DETECTION PROJECT")
    print("=" * 60)

    config_path = "configs/config.yaml"
    cfg = ConfigParser(config_path)

    # 1. Run Data Preprocessing Pipeline (Phase 2)
    print("\nRunning Feature Engineering and Preprocessing...")
    run_pipeline(config_path)

    # 2. Setup Central Logger for ML Experiment Results
    logger = ExperimentLogger(cfg.config, experiment_name="automata_experiment")

    datasets = ["swat", "wadi", "batadal"]
    results = []

    logger.start_timer("total_experiment_time")

    # 3. Train & Evaluate on each dataset
    for ds in datasets:
        try:
            res = run_experiment(cfg, logger, ds)
            results.append(res)
        except Exception as e:
            logger.error(f"Error running experiment on {ds}: {str(e)}")

    logger.stop_timer("total_experiment_time")
    logger.info("All experiments complete.")

    # Save metrics file
    logger.save_metrics(fmt="both")
    logger.summary()

    # 4. Final verification output
    print("\n" + "=" * 60)
    print("EXPERIMENT EXECUTION COMPLETE")
    print("=" * 60)
    for res in results:
        print(f"Dataset: {res['dataset'].upper():<10} | Test F1-Score: {res['test_f1']:.4f} | Accuracy: {res['test_acc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
