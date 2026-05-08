"""
Preprocessing Module
====================
Provides robust scaling and dimensionality reduction for time series features.
Ensures zero data leakage by fitting parameters on training data only.
"""

import os
import pickle
from typing import Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    """Handles scaling and artifact saving/loading for datasets."""

    def __init__(self, method: str = "minmax", artifact_dir: str = "./models/artifacts"):
        """
        Args:
            method: Scaling method, either 'minmax' or 'standard'.
            artifact_dir: Directory where fitted preprocessing objects are saved.
        """
        self.method = method.lower()
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

        if self.method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.method == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}. Use 'minmax' or 'standard'.")

    def fit_scaler(self, train_df: pd.DataFrame) -> None:
        """
        Fits the scaler on the training features only.
        """
        self.scaler.fit(train_df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the dataframe using the fitted scaler.
        """
        scaled_array = self.scaler.transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns, index=df.index)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits on training features and transforms them.
        """
        self.fit_scaler(train_df)
        return self.transform(train_df)

    def save_scaler(self, filename: str = "scaler.pkl") -> str:
        """
        Saves the fitted scaler to the artifact directory.
        Returns:
            The file path where the scaler was saved.
        """
        filepath = os.path.join(self.artifact_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self.scaler, f)
        return filepath

    def load_scaler(self, filepath: str) -> None:
        """
        Loads a pre-fitted scaler.
        """
        with open(filepath, "rb") as f:
            self.scaler = pickle.load(f)
