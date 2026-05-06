"""
Temporal Splitter Module
========================
Provides robust, chronological train/validation/test splitting for time series datasets.
Prevents data leakage by strictly maintaining temporal order (no random shuffling).
"""

import pandas as pd
from typing import Tuple


class TemporalSplitter:
    """Splits time series data chronologically."""

    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        """
        Args:
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(
        self, features: pd.DataFrame, times: pd.DataFrame = None
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.DataFrame, pd.DataFrame, pd.DataFrame
    ]:
        """
        Splits features (and optionally times) into Train, Val, Test chronologically.

        Returns:
            Tuple of:
            (train_features, val_features, test_features, train_times, val_times, test_times)
            If times is None, the time dataframes returned are empty.
        """
        n = len(features)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_features = features.iloc[:train_end].copy()
        val_features = features.iloc[train_end:val_end].copy()
        test_features = features.iloc[val_end:].copy()

        if times is not None and not times.empty:
            train_times = times.iloc[:train_end].copy()
            val_times = times.iloc[train_end:val_end].copy()
            test_times = times.iloc[val_end:].copy()
        else:
            train_times, val_times, test_times = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        return (
            train_features, val_features, test_features,
            train_times, val_times, test_times
        )

    def split_batadal(self, features: pd.DataFrame, times: pd.DataFrame) -> Tuple:
        """
        Splits BATADAL dataset specifically (60% Train, 20% Val, 20% Test).
        """
        # Ensure exact ratios as requested
        self.train_ratio, self.val_ratio, self.test_ratio = 0.6, 0.2, 0.2
        return self.split(features, times)

    def split_swat(self, features: pd.DataFrame, times: pd.DataFrame) -> Tuple:
        """
        Splits SWAT dataset chronologically.
        """
        return self.split(features, times)

    def split_wadi(self, features: pd.DataFrame, times: pd.DataFrame) -> Tuple:
        """
        Splits WADI dataset chronologically.
        """
        return self.split(features, times)
