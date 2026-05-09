"""
Pipeline Orchestration Script
==============================
Orchestrates the loading, splitting, scaling, PCA reduction,
and saving of processed datasets for SWAT, WADI, and BATADAL.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Tuple

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config_parser import ConfigParser
from src.utils.logger import ExperimentLogger
from src.data.data_loader import DataLoader
from src.data.splitter import TemporalSplitter
from src.data.preprocessor import DataPreprocessor, PCADimensionalityReducer


def create_dummy_dataset(filepath: str, num_rows: int = 1000, num_features: int = 10) -> None:
    """Helper to create dummy CSV files for SWAT, WADI, and BATADAL if they don't exist."""
    print(f"Creating dummy dataset at {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Generate dummy datetime index
    dates = pd.date_range(start="2026-05-01", periods=num_rows, freq="min")

    data = {
        "timestamp": dates if "batadal" not in filepath else dates,
    }
    if "batadal" in filepath.lower():
        data = {"DATETIME": dates}

    # Add numeric features
    for i in range(num_features):
        data[f"sensor_{i}"] = np.sin(np.linspace(0, 50, num_rows)) + np.random.normal(0, 0.1, num_rows)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def preprocess_dataset(
    dataset_name: str,
    loader: DataLoader,
    splitter: TemporalSplitter,
    config: ConfigParser,
    logger: ExperimentLogger,
    processed_dir: str
) -> None:
    """Preprocess a single dataset."""
    logger.info(f"Starting preprocessing pipeline for {dataset_name.upper()}")

    # 1. Load dataset
    logger.info(f"Loading raw data for {dataset_name}...")
    load_fn = getattr(loader, f"load_{dataset_name}")
    features_df, times_df = load_fn()

    logger.info(f"Loaded {dataset_name} with features shape {features_df.shape} and times shape {times_df.shape}")

    # 2. Split dataset chronologically
    logger.info(f"Splitting {dataset_name} chronologically...")
    split_fn = getattr(splitter, f"split_{dataset_name}")
    (
        train_feat, val_feat, test_feat,
        train_time, val_time, test_time
    ) = split_fn(features_df, times_df)

    logger.info(f"Split results: Train={train_feat.shape}, Val={val_feat.shape}, Test={test_feat.shape}")

    # 3. Scaling (Zero Leakage: Fit on Train ONLY)
    logger.info("Initializing scaling...")
    method = config.get("data.normalization", "minmax")
    preprocessor = DataPreprocessor(method=method, artifact_dir=config.get("paths.model_dir") + "/artifacts")

    logger.info(f"Fitting scaler on Train set...")
    train_feat_scaled = preprocessor.fit_transform(train_feat)
    val_feat_scaled = preprocessor.transform(val_feat)
    test_feat_scaled = preprocessor.transform(test_feat)

    # Save fitted scaler
    scaler_path = preprocessor.save_scaler(f"{dataset_name}_scaler.pkl")
    logger.info(f"Fitted scaler saved to {scaler_path}")

    # 4. PCA (Zero Leakage: Fit on Train ONLY)
    logger.info("Initializing PCA dimensionality reduction (1D)...")
    pca_reducer = PCADimensionalityReducer(n_components=1, artifact_dir=config.get("paths.model_dir") + "/artifacts")

    logger.info("Fitting PCA on Train set...")
    train_feat_pca = pca_reducer.fit_transform(train_feat_scaled)
    val_feat_pca = pca_reducer.transform(val_feat_scaled)
    test_feat_pca = pca_reducer.transform(test_feat_scaled)

    # Save fitted PCA
    pca_path = pca_reducer.save_pca(f"{dataset_name}_pca.pkl")
    logger.info(f"Fitted PCA saved to {pca_path}")

    # 5. Save Processed Datasets
    logger.info(f"Saving processed data to {processed_dir}...")
    dataset_out_dir = os.path.join(processed_dir, dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)

    # Save scaling features (CSV)
    train_feat_scaled.to_csv(os.path.join(dataset_out_dir, "train_scaled.csv"), index=False)
    val_feat_scaled.to_csv(os.path.join(dataset_out_dir, "val_scaled.csv"), index=False)
    test_feat_scaled.to_csv(os.path.join(dataset_out_dir, "test_scaled.csv"), index=False)

    # Save PCA features (1D) (Numpy & CSV)
    np.save(os.path.join(dataset_out_dir, "train_pca.npy"), train_feat_pca.values)
    np.save(os.path.join(dataset_out_dir, "val_pca.npy"), val_feat_pca.values)
    np.save(os.path.join(dataset_out_dir, "test_pca.npy"), test_feat_pca.values)

    train_feat_pca.to_csv(os.path.join(dataset_out_dir, "train_pca.csv"), index=False)
    val_feat_pca.to_csv(os.path.join(dataset_out_dir, "val_pca.csv"), index=False)
    test_feat_pca.to_csv(os.path.join(dataset_out_dir, "test_pca.csv"), index=False)

    # Save timestamps
    train_time.to_csv(os.path.join(dataset_out_dir, "train_time.csv"), index=False)
    val_time.to_csv(os.path.join(dataset_out_dir, "val_time.csv"), index=False)
    test_time.to_csv(os.path.join(dataset_out_dir, "test_time.csv"), index=False)

    logger.info(f"Finished preprocessing pipeline for {dataset_name.upper()}\n" + "-"*40)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    cfg = ConfigParser(args.config)

    # Setup logger
    logger = ExperimentLogger(cfg.config, experiment_name="data_preprocessing")

    data_dir = cfg.get("paths.data_dir")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Verify if raw datasets exist, otherwise generate dummy data for demonstration/sanity check
    datasets = ["swat", "wadi", "batadal"]
    for ds in datasets:
        ds_path = os.path.join(data_dir, f"{ds}.csv")
        if not os.path.exists(ds_path):
            logger.warning(f"Raw dataset {ds} not found at {ds_path}. Creating dummy data...")
            create_dummy_dataset(ds_path)

    # Initialize data tools
    loader = DataLoader(data_dir)
    # Get ratios from config, with fallback
    train_ratio = cfg.get("data.train_ratio", 0.6)
    val_ratio = cfg.get("data.val_ratio", 0.2)
    test_ratio = cfg.get("data.test_ratio", 0.2)
    splitter = TemporalSplitter(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

    logger.start_timer("total_preprocessing_time")

    # Run for all datasets
    for ds in datasets:
        try:
            preprocess_dataset(ds, loader, splitter, cfg, logger, processed_dir)
        except Exception as e:
            logger.error(f"Failed to preprocess {ds}: {str(e)}")

    logger.stop_timer("total_preprocessing_time")
    logger.info("Pipeline Execution Complete.")
    logger.summary()


if __name__ == "__main__":
    main()
