"""
Metrics Utilities
=================
Contains functions for aligning original time series labels with SAX pattern windows
and computing validation/test metrics.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any


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


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Computes Accuracy, Precision, Recall, and F1-score.

    Args:
        labels: 1D array of ground truth labels.
        predictions: 1D array of model predictions.

    Returns:
        Dictionary of computed metrics.
    """
    # Handle length mismatch due to padding/windowing edge cases
    min_len = min(len(labels), len(predictions))
    lbls = labels[:min_len]
    preds = predictions[:min_len]

    acc = accuracy_score(lbls, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(lbls, preds, average='binary', zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }
