"""
Dummy Dataset Generator
=======================
Generates synthetic SWAT, WADI, and BATADAL datasets for testing and demonstration.
Includes realistic timestamp alignments and z-score anomaly labels (~5%).
"""

import os
import numpy as np
import pandas as pd


def create_dummy_dataset(filepath: str, num_rows: int = 1000, num_features: int = 10) -> None:
    """
    Creates a synthetic z-normalized multivariate time-series CSV file with anomaly labels.

    Args:
        filepath: Target CSV file path.
        num_rows: Number of time-series observations.
        num_features: Number of sensor features.
    """
    print(f"Creating dummy dataset at {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Generate dummy datetime index
    dates = pd.date_range(start="2026-05-01", periods=num_rows, freq="min")

    data = {
        "timestamp": dates,
    }
    if "batadal" in filepath.lower():
        data = {"DATETIME": dates}

    # Add numeric features
    for i in range(num_features):
        data[f"sensor_{i}"] = np.sin(np.linspace(0, 50, num_rows)) + np.random.normal(0, 0.1, num_rows)

    # Add dummy label column with ~5% anomalies
    labels = np.random.choice([0, 1], size=num_rows, p=[0.95, 0.05])
    if "batadal" in filepath.lower():
        data["ATT_FLAG"] = labels
    else:
        data["Normal/Attack"] = ["Normal" if l == 0 else "Attack" for l in labels]

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
