"""
Data Loader Module
==================
Loads SKAB and BATADAL datasets from CSV formats.

Responsibilities:
- Load only the required datasets and files specified in the project document.
- Separate timestamps/metadata, labels, and model features.
- Convert dataset-specific anomaly labels into a standardized binary format:
    0 = normal
    1 = anomaly
- Prevent metadata and target columns from leaking into model inputs.
"""

import glob
import os
from typing import Dict, List, Tuple

import pandas as pd


class DataLoader:
    """Handles loading and separation of timestamps, labels, metadata, and features."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Removes leading/trailing spaces from column names."""
        df = df.copy()
        df.columns = df.columns.astype(str).str.strip()
        return df

    @staticmethod
    def _parse_binary_labels(
        labels: pd.Series,
        anomaly_values,
        normal_values,
        dataset_name: str,
        label_column: str
    ) -> pd.Series:
        """
        Converts raw dataset labels into standardized binary labels.

        Args:
            labels: Raw label column.
            anomaly_values: Values that should be mapped to 1.
            normal_values: Values that should be mapped to 0.
            dataset_name: Dataset name used in error messages.
            label_column: Original label column name.

        Returns:
            pd.Series containing only integer 0/1 labels.

        Raises:
            ValueError: If unexpected label values are found.
        """
        anomaly_set = {str(v).strip().lower() for v in anomaly_values}
        normal_set = {str(v).strip().lower() for v in normal_values}

        parsed = []
        unexpected_values = set()

        for raw_value in labels:
            value = str(raw_value).strip().lower()

            if value in anomaly_set:
                parsed.append(1)
            elif value in normal_set:
                parsed.append(0)
            else:
                unexpected_values.add(raw_value)

        if unexpected_values:
            raise ValueError(
                f"Unexpected label values found in {dataset_name}.{label_column}: "
                f"{sorted(unexpected_values)}. "
                f"Expected anomaly values={sorted(anomaly_set)} or "
                f"normal values={sorted(normal_set)}."
            )

        return pd.Series(parsed, index=labels.index, name="label", dtype="int64")

    @staticmethod
    def _ensure_numeric_features(features_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Converts feature columns to numeric values and handles invalid values safely.

        Non-numeric feature columns are converted to NaN. Columns that are entirely NaN
        are removed. Remaining missing values are filled with 0.
        """
        numeric_df = features_df.apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.dropna(axis=1, how="all")
        numeric_df = numeric_df.fillna(0)

        if numeric_df.empty:
            raise ValueError(f"No numeric feature columns found for {dataset_name}.")

        return numeric_df

    def load_skab(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the SKAB dataset by concatenating all CSV files in valve1 and valve2 folders.

        Required by the project:
        - Use only valve1 and valve2.
        - Add source_group and source_file columns.
        - Use anomaly as the target label.
        - Exclude datetime, changepoint, source_group, and source_file from model inputs.

        Returns:
            Tuple of (features_df, labels_df, metadata_df)
        """
        skab_dir = os.path.join(self.data_dir, "skab")

        if not os.path.exists(skab_dir):
            raise FileNotFoundError(f"SKAB dataset not found at {skab_dir}")

        all_dfs = []

        for valve in ["valve1", "valve2"]:
            valve_dir = os.path.join(skab_dir, valve)

            if not os.path.exists(valve_dir):
                raise FileNotFoundError(
                    f"Required SKAB folder not found: {valve_dir}. "
                    "The project requires both valve1 and valve2 folders."
                )

            csv_files = sorted(glob.glob(os.path.join(valve_dir, "*.csv")))

            if not csv_files:
                raise ValueError(f"No CSV files found in required SKAB folder: {valve_dir}")

            for file_path in csv_files:
                df = pd.read_csv(file_path, sep=";")
                df = self._standardize_columns(df)

                df["source_group"] = valve
                df["source_file"] = os.path.basename(file_path)

                all_dfs.append(df)

        full_df = pd.concat(all_dfs, ignore_index=True)

        if "anomaly" not in full_df.columns:
            raise ValueError("SKAB target label column 'anomaly' was not found.")

        metadata_cols = ["datetime", "changepoint", "source_group", "source_file"]
        actual_metadata_cols = [col for col in metadata_cols if col in full_df.columns]

        metadata_df = full_df[actual_metadata_cols].copy()
        labels_df = pd.DataFrame({
    "label": self._parse_binary_labels(
        labels=full_df["anomaly"],
        anomaly_values=[1, "1", "1.0", "anomaly", "attack", True],
        normal_values=[0, "0", "0.0", "normal", False],
        dataset_name="SKAB",
        label_column="anomaly"
    )
})

       

        drop_cols = actual_metadata_cols + ["anomaly"]
        features_df = full_df.drop(columns=drop_cols, errors="ignore")
        features_df = self._ensure_numeric_features(features_df, "SKAB")

        return features_df, labels_df, metadata_df

    def load_batadal(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the BATADAL Training Dataset 2 file.

        Required by the project:
        - Use only Training Dataset 2.
        - The local file is expected to be stored as data/batadal.csv.
        - Use ATT_FLAG as the target label.
        - Convert ATT_FLAG=1 to anomaly and ATT_FLAG=-999 to normal.
        - Exclude datetime/time columns from model inputs.

        Returns:
            Tuple of (features_df, labels_df, time_df)
        """
        filepath = os.path.join(self.data_dir, "batadal.csv")

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"BATADAL Training Dataset 2 file not found at {filepath}. "
                "Place the BATADAL Training Dataset 2 CSV as data/batadal.csv."
            )

        df = pd.read_csv(filepath)
        df = self._standardize_columns(df)

        if "ATT_FLAG" not in df.columns:
            raise ValueError(
                "BATADAL target label column 'ATT_FLAG' was not found. "
                "Make sure data/batadal.csv is the Training Dataset 2 file."
            )

        time_cols = ["DATETIME", "datetime", "Date", "Time", "timestamp"]
        actual_time_cols = [col for col in time_cols if col in df.columns]
        time_df = df[actual_time_cols].copy() if actual_time_cols else pd.DataFrame(index=df.index)

        labels_df = pd.DataFrame({
            "label": self._parse_binary_labels(
                labels=df["ATT_FLAG"],
                anomaly_values=[1, "1", "1.0", "attack", "anomaly", True],
                normal_values=[-999, "-999", "-999.0", 0, "0", "0.0", "normal", False],
                dataset_name="BATADAL",
                label_column="ATT_FLAG"
            )
        })

        drop_cols = actual_time_cols + ["ATT_FLAG"]
        features_df = df.drop(columns=drop_cols, errors="ignore")
        features_df = self._ensure_numeric_features(features_df, "BATADAL")

        return features_df, labels_df, time_df

    def get_dataset_info(self) -> Dict[str, Dict[str, object]]:
        """
        Returns basic dataset availability information for debugging and reporting.
        """
        return {
            "skab": {
                "expected_path": os.path.join(self.data_dir, "skab"),
                "required_groups": ["valve1", "valve2"],
                "target_column": "anomaly"
            },
            "batadal": {
                "expected_path": os.path.join(self.data_dir, "batadal.csv"),
                "expected_source": "BATADAL Training Dataset 2",
                "target_column": "ATT_FLAG"
            }
        }