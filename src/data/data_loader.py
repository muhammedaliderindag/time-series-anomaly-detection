"""
Data Loader Module
==================
Loads SWAT, WADI, and BATADAL datasets from CSV formats.
Ensures timestamp/datetime and anomaly label columns are correctly parsed,
separated from the feature sets, and returned.
"""

import os
import pandas as pd
from typing import Tuple, List, Optional


class DataLoader:
    """Handles loading and separation of timestamps, labels, and features."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to the root directory containing datasets.
        """
        self.data_dir = data_dir
        # Common column names for labels in these datasets
        self.label_cols = ["Normal/Attack", "Attack", "ATT_FLAG", "label", "Label", "Normal/Attack "]

    def _load_and_separate(
        self, filepath: str, time_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads a CSV file and separates the time and label columns from features.

        Args:
            filepath: Path to the CSV file.
            time_cols: List of column names to extract as timestamps.

        Returns:
            Tuple of (features_df, labels_df, time_df)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")

        # Basic loading
        df = pd.read_csv(filepath)

        # 1. Separate timestamp columns
        actual_time_cols = [col for col in time_cols if col in df.columns]
        time_df = df[actual_time_cols].copy() if actual_time_cols else pd.DataFrame()

        # 2. Separate label columns
        actual_label_cols = [col for col in self.label_cols if col in df.columns]
        
        if actual_label_cols:
            # Take the first matched label column
            lbl_col = actual_label_cols[0]
            labels_df = df[[lbl_col]].copy()
            # Rename label column to a standard 'label' name
            labels_df.columns = ['label']
            
            # Map text labels to binary if necessary (e.g. SWAT "Normal" -> 0, "Attack" -> 1)
            # Normalizing string/numerical values to standard 0/1 binary labels
            labels_df['label'] = labels_df['label'].apply(
                lambda x: 1 if str(x).strip().lower() in ['attack', '1', '1.0', 'anomaly'] else 0
            )
        else:
            labels_df = pd.DataFrame()

        # 3. Features are everything else
        drop_cols = actual_time_cols + actual_label_cols
        features_df = df.drop(columns=drop_cols, errors='ignore')

        return features_df, labels_df, time_df

    def load_swat(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the SWAT dataset.
        Returns:
            Tuple of (features_df, labels_df, time_df)
        """
        filepath = os.path.join(self.data_dir, "swat.csv")
        time_cols = ["Timestamp", "timestamp", "date", "Date"]
        return self._load_and_separate(filepath, time_cols)

    def load_wadi(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the WADI dataset.
        Returns:
            Tuple of (features_df, labels_df, time_df)
        """
        filepath = os.path.join(self.data_dir, "wadi.csv")
        time_cols = ["Row", "Date", "Time", "Date ", "Time ", "timestamp"]
        return self._load_and_separate(filepath, time_cols)

    def load_batadal(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the BATADAL dataset.
        Returns:
            Tuple of (features_df, labels_df, time_df)
        """
        filepath = os.path.join(self.data_dir, "batadal.csv")
        time_cols = ["DATETIME", "datetime", "Date", "Time", "timestamp"]
        return self._load_and_separate(filepath, time_cols)
