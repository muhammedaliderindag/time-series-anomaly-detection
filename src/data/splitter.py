"""
Temporal Splitter Module
========================
Provides robust, chronological train/validation/test splitting for time series datasets.
Prevents data leakage by strictly maintaining temporal order (no random shuffling).
Supports features, labels, and timestamps.
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
        self, features: pd.DataFrame, labels: pd.DataFrame = None, times: pd.DataFrame = None
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,  # features
        pd.DataFrame, pd.DataFrame, pd.DataFrame,  # labels
        pd.DataFrame, pd.DataFrame, pd.DataFrame   # times
    ]:
        """
        Splits features, labels, and times into Train, Val, Test chronologically.

        Returns:
            Tuple of:
            (train_feat, val_feat, test_feat, train_lbl, val_lbl, test_lbl, train_time, val_time, test_time)
        """
        n = len(features)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        # 1. Features
        train_feat = features.iloc[:train_end].copy()
        val_feat = features.iloc[train_end:val_end].copy()
        test_feat = features.iloc[val_end:].copy()

        # 2. Labels
        if labels is not None and not labels.empty:
            train_lbl = labels.iloc[:train_end].copy()
            val_lbl = labels.iloc[train_end:val_end].copy()
            test_lbl = labels.iloc[val_end:].copy()
        else:
            train_lbl, val_lbl, test_lbl = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # 3. Timestamps
        if times is not None and not times.empty:
            train_time = times.iloc[:train_end].copy()
            val_time = times.iloc[train_end:val_end].copy()
            test_time = times.iloc[val_end:].copy()
        else:
            train_time, val_time, test_time = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        return (
            train_feat, val_feat, test_feat,
            train_lbl, val_lbl, test_lbl,
            train_time, val_time, test_time
        )

    def split_batadal(self, features: pd.DataFrame, labels: pd.DataFrame, times: pd.DataFrame) -> Tuple:
        """
        Splits BATADAL dataset specifically (60% Train, 20% Val, 20% Test).
        """
        self.train_ratio, self.val_ratio, self.test_ratio = 0.6, 0.2, 0.2
        return self.split(features, labels, times)

    def split_swat(self, features: pd.DataFrame, labels: pd.DataFrame, times: pd.DataFrame) -> Tuple:
        """
        Splits SWAT dataset chronologically.
        """
        return self.split(features, labels, times)

    def split_wadi(self, features: pd.DataFrame, labels: pd.DataFrame, times: pd.DataFrame) -> Tuple:
        """
        Splits WADI dataset chronologically.
        """
        return self.split(features, labels, times)
