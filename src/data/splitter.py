"""
Data Splitter Module
====================
Provides dataset-specific splitting strategies ensuring no data leakage.
- BATADAL: Chronological 60/20/20 split.
- SKAB: GroupKFold based on source_file, ensuring the same file doesn't appear
  in both train and test simultaneously.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from typing import Tuple, List, Dict


class DatasetSplitter:
    """Handles dataset-specific splitting logic."""

    def split_batadal(self, features: pd.DataFrame, labels: pd.DataFrame, times: pd.DataFrame) -> Tuple:
        """
        Splits BATADAL dataset chronologically (60% Train, 20% Val, 20% Test).
        """
        n = len(features)
        train_end = int(n * 0.6)
        val_end = train_end + int(n * 0.2)

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

    def get_skab_folds(self, features: pd.DataFrame, labels: pd.DataFrame, meta: pd.DataFrame, n_splits: int = 5) -> List[Dict[str, Tuple]]:
        """
        Generates K-Folds for SKAB using GroupKFold on the 'source_file' column.
        Splits the train_val set further into train (80%) and val (20%) chronologically
        within the fold to maintain temporal integrity.
        
        Returns:
            List of dictionaries containing the fold data:
            [
                {
                    "train": (train_feat, train_lbl, train_meta),
                    "val": (val_feat, val_lbl, val_meta),
                    "test": (test_feat, test_lbl, test_meta)
                }, ...
            ]
        """
        gkf = GroupKFold(n_splits=n_splits)
        groups = meta["source_file"].values
        folds = []

        # If there are fewer groups than n_splits, adjust n_splits
        unique_groups = len(np.unique(groups))
        if unique_groups < n_splits:
            n_splits = unique_groups
            gkf = GroupKFold(n_splits=n_splits)

        for train_val_idx, test_idx in gkf.split(features, groups=groups):
            # Sort the indices to maintain chronological order
            train_val_idx = np.sort(train_val_idx)
            test_idx = np.sort(test_idx)
            
            # Split train_val into train and val (80/20)
            tv_len = len(train_val_idx)
            val_size = int(tv_len * 0.2)
            train_idx = train_val_idx[:-val_size]
            val_idx = train_val_idx[-val_size:]

            fold_data = {
                "train": (
                    features.iloc[train_idx].copy(),
                    labels.iloc[train_idx].copy() if not labels.empty else pd.DataFrame(),
                    meta.iloc[train_idx].copy()
                ),
                "val": (
                    features.iloc[val_idx].copy(),
                    labels.iloc[val_idx].copy() if not labels.empty else pd.DataFrame(),
                    meta.iloc[val_idx].copy()
                ),
                "test": (
                    features.iloc[test_idx].copy(),
                    labels.iloc[test_idx].copy() if not labels.empty else pd.DataFrame(),
                    meta.iloc[test_idx].copy()
                )
            }
            folds.append(fold_data)

        return folds
