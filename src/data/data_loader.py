"""
Data Loader Module
==================
Loads SWAT, WADI, and BATADAL datasets from CSV/Numpy formats.
Ensures timestamp/datetime columns are separated from the feature sets.
"""

import os
import pandas as pd
from typing import Tuple, List, Optional


class DataLoader:
    """Handles loading and basic timestamp separation for time-series datasets."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to the root directory containing datasets.
        """
        self.data_dir = data_dir

    def _load_and_separate_time(
        self, filepath: str, time_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads a CSV file and separates the time columns from the features.

        Args:
            filepath: Path to the CSV file.
            time_cols: List of column names to extract as timestamps.

        Returns:
            Tuple of (features_df, time_df)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")

        # Basic loading, we assume CSV for these public datasets unless specified otherwise.
        df = pd.read_csv(filepath)

        # Identify which time columns actually exist in the dataframe
        actual_time_cols = [col for col in time_cols if col in df.columns]

        time_df = df[actual_time_cols].copy() if actual_time_cols else pd.DataFrame()
        features_df = df.drop(columns=actual_time_cols)

        return features_df, time_df

    def load_swat(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the SWAT dataset.
        Returns:
            Tuple of (features_df, time_df)
        """
        filepath = os.path.join(self.data_dir, "swat.csv")
        # SWAT typically uses 'Timestamp' or 'date'
        time_cols = ["Timestamp", "timestamp", "date", "Date"]
        return self._load_and_separate_time(filepath, time_cols)

    def load_wadi(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the WADI dataset.
        Returns:
            Tuple of (features_df, time_df)
        """
        filepath = os.path.join(self.data_dir, "wadi.csv")
        # WADI typically uses 'Row' or 'Date' / 'Time'
        time_cols = ["Row", "Date", "Time", "Date ", "Time ", "timestamp"]
        return self._load_and_separate_time(filepath, time_cols)

    def load_batadal(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the BATADAL dataset.
        Returns:
            Tuple of (features_df, time_df)
        """
        filepath = os.path.join(self.data_dir, "batadal.csv")
        # BATADAL typically uses 'DATETIME'
        time_cols = ["DATETIME", "datetime", "Date", "Time", "timestamp"]
        return self._load_and_separate_time(filepath, time_cols)
