"""
Data Loader Module
==================
Loads SKAB and BATADAL datasets from CSV formats.
Ensures timestamp, label, source_group, source_file columns are correctly parsed,
separated from the feature sets, and returned.
"""

import os
import glob
from typing import Tuple

import pandas as pd


class DataLoader:
    """Handles loading and separation of timestamps, labels, metadata, and features."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_skab(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the SKAB dataset by concatenating all CSV files in valve1 and valve2 folders.

        Adds:
            source_group: valve1 or valve2
            source_file: original CSV filename

        Returns:
            Tuple of (features_df, labels_df, metadata_df)

            features_df:
                Numeric sensor features.

            labels_df:
                Binary anomaly labels from the anomaly column.

            metadata_df:
                datetime, changepoint, source_group, source_file columns if available.
        """
        skab_dir = os.path.join(self.data_dir, "skab")

        if not os.path.exists(skab_dir):
            raise FileNotFoundError(f"SKAB dataset not found at {skab_dir}")

        all_dfs = []

        for valve in ["valve1", "valve2"]:
            valve_dir = os.path.join(skab_dir, valve)

            if not os.path.exists(valve_dir):
                continue

            csv_files = glob.glob(os.path.join(valve_dir, "*.csv"))

            for file_path in csv_files:
                df = pd.read_csv(file_path, sep=";")
                df.columns = df.columns.str.strip()

                df["source_group"] = valve
                df["source_file"] = os.path.basename(file_path)

                all_dfs.append(df)

        if not all_dfs:
            raise ValueError("No CSV files found in SKAB valve1/valve2 directories.")

        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df.columns = full_df.columns.str.strip()

        meta_cols = ["datetime", "changepoint", "source_group", "source_file"]
        actual_meta_cols = [col for col in meta_cols if col in full_df.columns]

        metadata_df = (
            full_df[actual_meta_cols].copy()
            if actual_meta_cols
            else pd.DataFrame(index=full_df.index)
        )

        if "anomaly" not in full_df.columns:
            raise ValueError("SKAB label column 'anomaly' not found.")

        labels_df = full_df[["anomaly"]].copy()
        labels_df.columns = ["label"]
        labels_df["label"] = labels_df["label"].apply(lambda x: 1 if int(x) == 1 else 0)

        drop_cols = actual_meta_cols + ["anomaly"]
        features_df = full_df.drop(columns=drop_cols, errors="ignore")

        features_df = features_df.apply(pd.to_numeric, errors="coerce")
        features_df = features_df.dropna(axis=1, how="all")
        features_df = features_df.fillna(0)

        return features_df, labels_df, metadata_df

    def load_batadal(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the BATADAL Training Dataset 2.

        ATT_FLAG interpretation:
            1    -> attack/anomaly
            -999 -> unlabeled/background, treated as normal class 0 for binary evaluation

        Returns:
            Tuple of (features_df, labels_df, time_df)

            features_df:
                Numeric sensor/actuator features.

            labels_df:
                Binary labels where 1 is anomaly and 0 is normal/background.

            time_df:
                Timestamp column if available.
        """
        filepath = os.path.join(self.data_dir, "batadal.csv")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"BATADAL dataset not found at {filepath}")

        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        time_cols = ["DATETIME", "datetime", "Date", "Time", "timestamp"]
        actual_time_cols = [col for col in time_cols if col in df.columns]

        time_df = (
            df[actual_time_cols].copy()
            if actual_time_cols
            else pd.DataFrame(index=df.index)
        )

        if "ATT_FLAG" not in df.columns:
            raise ValueError("BATADAL label column 'ATT_FLAG' not found.")

        labels_df = df[["ATT_FLAG"]].copy()
        labels_df.columns = ["label"]

        # BATADAL Training Dataset 2 is partially labeled:
        # 1 = attack/anomaly, -999 = unlabeled/background.
        # For binary evaluation, -999 is treated as class 0.
        labels_df["label"] = labels_df["label"].apply(lambda x: 1 if int(x) == 1 else 0)

        drop_cols = actual_time_cols + ["ATT_FLAG"]
        features_df = df.drop(columns=drop_cols, errors="ignore")

        features_df = features_df.apply(pd.to_numeric, errors="coerce")
        features_df = features_df.dropna(axis=1, how="all")
        features_df = features_df.fillna(0)

        return features_df, labels_df, time_df