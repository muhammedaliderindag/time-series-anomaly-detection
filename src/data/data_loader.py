"""
Data Loader Module
==================
Loads SKAB and BATADAL datasets from CSV formats.
Ensures timestamp, label, source_group, source_file columns are correctly parsed,
separated from the feature sets, and returned.
"""

import os
import glob
import pandas as pd
from typing import Tuple, List

class DataLoader:
    """Handles loading and separation of timestamps, labels, and features."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_skab(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the SKAB dataset by concatenating all CSVs in valve1 and valve2 folders.
        Adds 'source_group' and 'source_file' columns.
        Returns:
            Tuple of (features_df, labels_df, metadata_df)
            metadata_df contains datetime, changepoint, source_group, source_file.
        """
        skab_dir = os.path.join(self.data_dir, "skab")
        if not os.path.exists(skab_dir):
            raise FileNotFoundError(f"SKAB dataset not found at {skab_dir}")

        all_dfs = []
        for valve in ["valve1", "valve2"]:
            valve_dir = os.path.join(skab_dir, valve)
            if not os.path.exists(valve_dir):
                continue
            
            for file_path in glob.glob(os.path.join(valve_dir, "*.csv")):
                df = pd.read_csv(file_path)
                df["source_group"] = valve
                df["source_file"] = os.path.basename(file_path)
                all_dfs.append(df)

        if not all_dfs:
            raise ValueError("No CSV files found in SKAB directories.")

        full_df = pd.concat(all_dfs, ignore_index=True)

        # 1. Separate metadata columns
        meta_cols = ["datetime", "changepoint", "source_group", "source_file"]
        actual_meta_cols = [col for col in meta_cols if col in full_df.columns]
        meta_df = full_df[actual_meta_cols].copy()

        # 2. Separate label column
        if "anomaly" in full_df.columns:
            labels_df = full_df[["anomaly"]].copy()
            labels_df.columns = ["label"]
        else:
            labels_df = pd.DataFrame()

        # 3. Features are everything else
        drop_cols = actual_meta_cols + (["anomaly"] if "anomaly" in full_df.columns else [])
        features_df = full_df.drop(columns=drop_cols, errors="ignore")

        return features_df, labels_df, meta_df

    def load_batadal(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the BATADAL dataset.
        Returns:
            Tuple of (features_df, labels_df, time_df)
        """
        filepath = os.path.join(self.data_dir, "batadal.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")

        df = pd.read_csv(filepath)

        time_cols = ["DATETIME", "datetime", "Date", "Time", "timestamp"]
        actual_time_cols = [col for col in time_cols if col in df.columns]
        time_df = df[actual_time_cols].copy() if actual_time_cols else pd.DataFrame()

        if "ATT_FLAG" in df.columns:
            labels_df = df[["ATT_FLAG"]].copy()
            labels_df.columns = ["label"]
        else:
            labels_df = pd.DataFrame()

        drop_cols = actual_time_cols + (["ATT_FLAG"] if "ATT_FLAG" in df.columns else [])
        features_df = df.drop(columns=drop_cols, errors='ignore')

        return features_df, labels_df, time_df
