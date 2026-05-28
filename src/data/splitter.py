"""
Data Splitter Module
====================
Provides dataset-specific splitting strategies while preventing data leakage.

Project requirements:
- BATADAL: Chronological 60/20/20 split. No random row-level split.
- SKAB: GroupKFold based on source_file. The same CSV file must not appear
  in train and test simultaneously.

Additional robustness:
- SKAB validation split is also performed at source_file group level.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


class DatasetSplitter:
    """Handles dataset-specific splitting logic."""

    @staticmethod
    def _validate_equal_lengths(*dfs: pd.DataFrame) -> None:
        """Ensures all non-empty dataframes have the same number of rows."""
        lengths = [len(df) for df in dfs if df is not None and not df.empty]

        if lengths and len(set(lengths)) != 1:
            raise ValueError(f"Input dataframes have inconsistent lengths: {lengths}")

    def split_batadal(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        times: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits BATADAL chronologically into 60% train, 20% validation, and 20% test.

        This method preserves time order and does not perform random row-level splitting.
        """
        self._validate_equal_lengths(features, labels, times)

        n = len(features)

        if n < 5:
            raise ValueError("BATADAL dataset is too small for a 60/20/20 split.")

        train_end = int(n * 0.6)
        val_end = train_end + int(n * 0.2)

        train_feat = features.iloc[:train_end].copy()
        val_feat = features.iloc[train_end:val_end].copy()
        test_feat = features.iloc[val_end:].copy()

        if labels is not None and not labels.empty:
            train_lbl = labels.iloc[:train_end].copy()
            val_lbl = labels.iloc[train_end:val_end].copy()
            test_lbl = labels.iloc[val_end:].copy()
        else:
            train_lbl, val_lbl, test_lbl = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        if times is not None and not times.empty:
            train_time = times.iloc[:train_end].copy()
            val_time = times.iloc[train_end:val_end].copy()
            test_time = times.iloc[val_end:].copy()
        else:
            train_time = pd.DataFrame(index=train_feat.index)
            val_time = pd.DataFrame(index=val_feat.index)
            test_time = pd.DataFrame(index=test_feat.index)

        return (
            train_feat, val_feat, test_feat,
            train_lbl, val_lbl, test_lbl,
            train_time, val_time, test_time
        )

    @staticmethod
    def _split_train_val_groups(
        train_val_idx: np.ndarray,
        meta: pd.DataFrame,
        val_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits train_val indices into train and validation sets using source_file groups.

        The same source_file is never placed in both train and validation.
        Validation files are selected from the end of the ordered group list to preserve
        a deterministic and chronological-like split at file level.
        """
        train_val_meta = meta.iloc[train_val_idx].copy()

        ordered_groups = train_val_meta["source_file"].drop_duplicates().tolist()

        if len(ordered_groups) < 2:
            raise ValueError(
                "At least two SKAB source_file groups are required to create "
                "separate train and validation sets."
            )

        val_group_count = max(1, int(round(len(ordered_groups) * val_ratio)))

        if val_group_count >= len(ordered_groups):
            val_group_count = len(ordered_groups) - 1

        val_groups = set(ordered_groups[-val_group_count:])
        train_groups = set(ordered_groups[:-val_group_count])

        train_mask = train_val_meta["source_file"].isin(train_groups).values
        val_mask = train_val_meta["source_file"].isin(val_groups).values

        train_idx = train_val_idx[train_mask]
        val_idx = train_val_idx[val_mask]

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError("SKAB train/validation split produced an empty subset.")

        return np.sort(train_idx), np.sort(val_idx)

    def get_skab_folds(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        meta: pd.DataFrame,
        n_splits: int = 5
    ) -> List[Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]]:
        """
        Generates SKAB folds using GroupKFold on source_file.

        Guarantees:
        - Test files are not present in train/validation.
        - Validation files are not present in train.
        - source_file is used as the group variable.
        """
        self._validate_equal_lengths(features, labels, meta)

        if "source_file" not in meta.columns:
            raise ValueError("SKAB metadata must contain a 'source_file' column.")

        groups = meta["source_file"].values
        unique_groups = np.unique(groups)

        if len(unique_groups) < 2:
            raise ValueError("At least two unique SKAB source_file groups are required.")

        effective_splits = min(n_splits, len(unique_groups))

        if effective_splits < 2:
            raise ValueError("GroupKFold requires at least two splits.")

        gkf = GroupKFold(n_splits=effective_splits)
        folds = []

        for fold_id, (train_val_idx, test_idx) in enumerate(gkf.split(features, labels, groups)):
            train_val_idx = np.sort(train_val_idx)
            test_idx = np.sort(test_idx)

            train_idx, val_idx = self._split_train_val_groups(
                train_val_idx=train_val_idx,
                meta=meta,
                val_ratio=0.2
            )

            train_files = set(meta.iloc[train_idx]["source_file"].unique())
            val_files = set(meta.iloc[val_idx]["source_file"].unique())
            test_files = set(meta.iloc[test_idx]["source_file"].unique())

            if train_files & val_files:
                raise ValueError(f"Leakage detected in fold {fold_id}: train and val share source_file.")
            if train_files & test_files:
                raise ValueError(f"Leakage detected in fold {fold_id}: train and test share source_file.")
            if val_files & test_files:
                raise ValueError(f"Leakage detected in fold {fold_id}: val and test share source_file.")

            fold_data = {
                "train": (
                    features.iloc[train_idx].copy(),
                    labels.iloc[train_idx].copy() if labels is not None and not labels.empty else pd.DataFrame(),
                    meta.iloc[train_idx].copy()
                ),
                "val": (
                    features.iloc[val_idx].copy(),
                    labels.iloc[val_idx].copy() if labels is not None and not labels.empty else pd.DataFrame(),
                    meta.iloc[val_idx].copy()
                ),
                "test": (
                    features.iloc[test_idx].copy(),
                    labels.iloc[test_idx].copy() if labels is not None and not labels.empty else pd.DataFrame(),
                    meta.iloc[test_idx].copy()
                )
            }

            folds.append(fold_data)

        return folds