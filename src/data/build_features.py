"""
Pipeline Orchestration Script
==============================
Orchestrates the loading, splitting, scaling, PCA reduction,
and saving of processed datasets for SKAB and BATADAL.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config_parser import ConfigParser
from src.utils.logger import ExperimentLogger
from src.data.data_loader import DataLoader
from src.data.splitter import DatasetSplitter
from src.data.preprocessor import DataPreprocessor, PCADimensionalityReducer
from src.data.dummy_generator import create_skab_dummy, create_batadal_dummy

def preprocess_and_save(
    fold_dir: str,
    train_feat: pd.DataFrame, val_feat: pd.DataFrame, test_feat: pd.DataFrame,
    train_lbl: pd.DataFrame, val_lbl: pd.DataFrame, test_lbl: pd.DataFrame,
    train_time: pd.DataFrame, val_time: pd.DataFrame, test_time: pd.DataFrame,
    config: ConfigParser, logger: ExperimentLogger, dataset_name: str
):
    os.makedirs(fold_dir, exist_ok=True)
    
    # 3. Scaling (Zero Leakage: Fit on Train ONLY)
    method = config.get("data.normalization", "minmax")
    preprocessor = DataPreprocessor(method=method, artifact_dir=fold_dir)
    train_feat_scaled = preprocessor.fit_transform(train_feat)
    val_feat_scaled = preprocessor.transform(val_feat)
    test_feat_scaled = preprocessor.transform(test_feat)
    preprocessor.save_scaler(f"{dataset_name}_scaler.pkl")

    # 4. PCA (Zero Leakage: Fit on Train ONLY)
    pca_reducer = PCADimensionalityReducer(n_components=1, artifact_dir=fold_dir)
    train_feat_pca = pca_reducer.fit_transform(train_feat_scaled)
    val_feat_pca = pca_reducer.transform(val_feat_scaled)
    test_feat_pca = pca_reducer.transform(test_feat_scaled)
    pca_reducer.save_pca(f"{dataset_name}_pca.pkl")

    # Save PCA features (1D) (Numpy)
    np.save(os.path.join(fold_dir, "train_pca.npy"), train_feat_pca.values)
    np.save(os.path.join(fold_dir, "val_pca.npy"), val_feat_pca.values)
    np.save(os.path.join(fold_dir, "test_pca.npy"), test_feat_pca.values)

    # Save labels
    if not train_lbl.empty: train_lbl.to_csv(os.path.join(fold_dir, "train_labels.csv"), index=False)
    if not val_lbl.empty: val_lbl.to_csv(os.path.join(fold_dir, "val_labels.csv"), index=False)
    if not test_lbl.empty: test_lbl.to_csv(os.path.join(fold_dir, "test_labels.csv"), index=False)

    # Save timestamps
    if not train_time.empty: train_time.to_csv(os.path.join(fold_dir, "train_time.csv"), index=False)
    if not val_time.empty: val_time.to_csv(os.path.join(fold_dir, "val_time.csv"), index=False)
    if not test_time.empty: test_time.to_csv(os.path.join(fold_dir, "test_time.csv"), index=False)

def preprocess_skab(loader: DataLoader, splitter: DatasetSplitter, config: ConfigParser, logger: ExperimentLogger, processed_dir: str):
    logger.info("Starting preprocessing pipeline for SKAB (K-Fold)")
    features_df, labels_df, meta_df = loader.load_skab()
    folds = splitter.get_skab_folds(features_df, labels_df, meta_df, n_splits=5)
    
    dataset_out_dir = os.path.join(processed_dir, "skab")
    
    for i, fold_data in enumerate(folds):
        logger.info(f"Processing SKAB Fold {i}...")
        fold_dir = os.path.join(dataset_out_dir, f"fold_{i}")
        t_f, t_l, t_m = fold_data["train"]
        v_f, v_l, v_m = fold_data["val"]
        te_f, te_l, te_m = fold_data["test"]
        preprocess_and_save(fold_dir, t_f, v_f, te_f, t_l, v_l, te_l, t_m, v_m, te_m, config, logger, "skab")

def preprocess_batadal(loader: DataLoader, splitter: DatasetSplitter, config: ConfigParser, logger: ExperimentLogger, processed_dir: str):
    logger.info("Starting preprocessing pipeline for BATADAL")
    features_df, labels_df, times_df = loader.load_batadal()
    (t_f, v_f, te_f, t_l, v_l, te_l, t_m, v_m, te_m) = splitter.split_batadal(features_df, labels_df, times_df)
    
    dataset_out_dir = os.path.join(processed_dir, "batadal")
    preprocess_and_save(dataset_out_dir, t_f, v_f, te_f, t_l, v_l, te_l, t_m, v_m, te_m, config, logger, "batadal")

def run_pipeline(config_path: str = "configs/config.yaml") -> None:
    cfg = ConfigParser(config_path)
    logger = ExperimentLogger(cfg.config, experiment_name="data_preprocessing")
    data_dir = cfg.get("paths.data_dir", "data")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Ensure dummies
    if not os.path.exists(os.path.join(data_dir, "skab")):
        logger.warning("SKAB dummy data not found. Creating...")
        create_skab_dummy(data_dir)
    if not os.path.exists(os.path.join(data_dir, "batadal.csv")):
        logger.warning("BATADAL dummy data not found. Creating...")
        create_batadal_dummy(os.path.join(data_dir, "batadal.csv"))

    loader = DataLoader(data_dir)
    splitter = DatasetSplitter()
    
    logger.start_timer("total_preprocessing_time")
    
    try:
        preprocess_skab(loader, splitter, cfg, logger, processed_dir)
    except Exception as e:
        logger.error(f"Failed to preprocess SKAB: {str(e)}")

    try:
        preprocess_batadal(loader, splitter, cfg, logger, processed_dir)
    except Exception as e:
        logger.error(f"Failed to preprocess BATADAL: {str(e)}")

    logger.stop_timer("total_preprocessing_time")
    logger.info("Pipeline Execution Complete.")
    logger.summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Preprocessing Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    run_pipeline(args.config)
