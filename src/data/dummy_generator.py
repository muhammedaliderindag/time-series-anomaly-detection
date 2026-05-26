"""
Dummy Dataset Generator
=======================
Generates synthetic SKAB and BATADAL datasets for testing and demonstration.
Includes realistic timestamp alignments and anomaly labels.
"""

import os
import numpy as np
import pandas as pd


def create_skab_dummy(data_dir: str, num_files_per_valve: int = 4, num_rows_per_file: int = 200, num_features: int = 8) -> None:
    """
    Creates the SKAB dummy dataset structure with valve1 and valve2 folders.
    Each folder contains multiple CSV files with datetime, anomaly, changepoint columns.
    """
    skab_dir = os.path.join(data_dir, "skab")
    valves = ["valve1", "valve2"]
    
    for valve in valves:
        valve_dir = os.path.join(skab_dir, valve)
        os.makedirs(valve_dir, exist_ok=True)
        
        for file_idx in range(num_files_per_valve):
            filepath = os.path.join(valve_dir, f"{file_idx}.csv")
            dates = pd.date_range(start=f"2026-05-{file_idx+1:02d}", periods=num_rows_per_file, freq="min")
            
            data = {
                "datetime": dates,
            }
            # Add numeric features
            for i in range(num_features):
                data[f"sensor_{i}"] = np.sin(np.linspace(0, 20, num_rows_per_file)) + np.random.normal(0, 0.1, num_rows_per_file)
            
            # Anomaly and changepoint
            labels = np.random.choice([0, 1], size=num_rows_per_file, p=[0.95, 0.05])
            data["anomaly"] = labels
            data["changepoint"] = [1 if (i > 0 and labels[i] != labels[i-1]) else 0 for i in range(num_rows_per_file)]
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
    print(f"Created SKAB dummy dataset at {skab_dir}")

def create_batadal_dummy(filepath: str, num_rows: int = 1000, num_features: int = 10) -> None:
    """
    Creates BATADAL (Training Dataset 2) dummy data.
    """
    print(f"Creating BATADAL dummy dataset at {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    dates = pd.date_range(start="2026-05-01", periods=num_rows, freq="min")
    data = {"DATETIME": dates}

    for i in range(num_features):
        data[f"sensor_{i}"] = np.sin(np.linspace(0, 50, num_rows)) + np.random.normal(0, 0.1, num_rows)

    labels = np.random.choice([0, 1], size=num_rows, p=[0.95, 0.05])
    data["ATT_FLAG"] = labels

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    create_skab_dummy("data")
    create_batadal_dummy("data/batadal.csv")
