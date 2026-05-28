"""
Deep Learning Experiment Module
===============================
Runs LSTM and 1D-CNN autoencoder experiments on processed PCA time-series data.

Outputs:
- results/dl_experiment_results.csv
- results/dl_experiment_results.json
- results/dl_experiment_summary.csv
- results/dl_experiment_summary.json

The module evaluates reconstruction-error-based anomaly detection using:
- dynamic threshold from train reconstruction scores
- test reconstruction scores for anomaly predictions
"""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.dl.architectures import CNNAutoencoder, LSTMAutoencoder
from src.models.dl.data_loader import create_dataloader
from src.models.dl.evaluate import (
    calculate_dynamic_threshold,
    detect_anomalies,
    evaluate_model,
)
from src.models.dl.trainer import ModelTrainer
from src.utils.config_parser import ConfigParser
from src.utils.metrics import calculate_metrics


class DeepLearningExperimentRunner:
    """Runs deep learning experiments for LSTM and 1D-CNN autoencoders."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = ConfigParser(config_path)
        self.config = self.cfg.config

        self.data_dir = self.cfg.get("paths.data_dir", "./data")
        self.results_dir = self.cfg.get("paths.results_dir", "./results")
        self.log_dir = self.cfg.get("paths.log_dir", "./logs")
        self.model_dir = self.cfg.get("paths.model_dir", "./models")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.sequence_length = self.cfg.get("data.sequence_length", 100)
        self.batch_size = self.cfg.get("batch_size", 32)
        self.seeds = self.cfg.get("random_seeds", [42, 123, 2026, 7, 999])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_seed(self, seed: int) -> None:
        """Sets random seeds for reproducible DL experiments."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_labels(self, path: str, length: int) -> np.ndarray:
        """Loads label CSV if available; otherwise returns zeros."""
        if os.path.exists(path):
            return pd.read_csv(path)["label"].values

        return np.zeros(length)

    def _labels_to_windows(self, labels: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Converts point-level labels to window-level labels.

        A window is anomalous if at least one timestamp inside that window is anomalous.
        """
        if len(labels) < sequence_length:
            return np.array([], dtype=int)

        window_labels = []

        for i in range(len(labels) - sequence_length + 1):
            window = labels[i:i + sequence_length]
            window_labels.append(1 if np.any(window == 1) else 0)

        return np.array(window_labels, dtype=int)

    def _get_dataset_dirs(self, dataset_name: str) -> List[Tuple[str, int]]:
        """Returns processed directories for BATADAL or SKAB folds."""
        processed_dir = os.path.join(self.data_dir, "processed", dataset_name)

        if dataset_name == "skab":
            fold_dirs = sorted([
                item for item in os.listdir(processed_dir)
                if item.startswith("fold_")
            ])

            return [
                (os.path.join(processed_dir, fold_dir), idx)
                for idx, fold_dir in enumerate(fold_dirs)
            ]

        return [(processed_dir, 0)]

    def _build_model(self, model_name: str) -> torch.nn.Module:
        """Creates a DL autoencoder model."""
        input_dim = self.cfg.get("model.input_dim", 1)
        hidden_dim = self.cfg.get("model.hidden_dim", 64)
        latent_dim = self.cfg.get("model.latent_dim", 32)
        num_layers = self.cfg.get("model.num_layers", 2)
        dropout = self.cfg.get("model.dropout", 0.2)

        if model_name == "lstm":
            return LSTMAutoencoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                num_layers=num_layers,
                dropout=dropout
            )

        if model_name == "cnn":
            return CNNAutoencoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                sequence_length=self.sequence_length
            )

        raise ValueError(f"Unsupported model_name: {model_name}")

    def _evaluate_single_fold(
        self,
        dataset_name: str,
        fold_dir: str,
        fold_idx: int,
        model_name: str,
        seed: int
    ) -> Dict[str, Any]:
        """Trains and evaluates one DL model on one dataset/fold."""
        train_pca = np.load(os.path.join(fold_dir, "train_pca.npy")).flatten()
        val_pca = np.load(os.path.join(fold_dir, "val_pca.npy")).flatten()
        test_pca = np.load(os.path.join(fold_dir, "test_pca.npy")).flatten()

        test_labels_raw = self._load_labels(
            os.path.join(fold_dir, "test_labels.csv"),
            len(test_pca)
        )

        train_loader = create_dataloader(
            train_pca,
            self.sequence_length,
            self.batch_size,
            shuffle=True
        )
        val_loader = create_dataloader(
            val_pca,
            self.sequence_length,
            self.batch_size,
            shuffle=False
        )
        test_loader = create_dataloader(
            test_pca,
            self.sequence_length,
            self.batch_size,
            shuffle=False
        )

        model = self._build_model(model_name)

        # Avoid checkpoint collisions between dataset/model/seed/fold runs.
        self.config.setdefault("paths", {})
        self.config["paths"]["model_dir"] = self.model_dir

        trainer = ModelTrainer(model, self.config, self.device)
        trainer.checkpoint_path = os.path.join(
            self.model_dir,
            f"best_{model_name}_{dataset_name}_seed{seed}_fold{fold_idx}.pt"
        )
        trainer.early_stopping.path = trainer.checkpoint_path

        train_start = time.perf_counter()
        train_losses, val_losses = trainer.train(train_loader, val_loader)
        training_time = time.perf_counter() - train_start

        inference_start = time.perf_counter()

        train_scores = evaluate_model(trainer.model, train_loader, self.device)
        test_scores = evaluate_model(trainer.model, test_loader, self.device)

        threshold_percentile = self.cfg.get(
            "deep_learning.threshold_percentile",
            95.0
        )
        threshold = calculate_dynamic_threshold(
            train_scores,
            percentile=threshold_percentile
        )

        test_preds = detect_anomalies(test_scores, threshold).astype(int)
        inference_time = time.perf_counter() - inference_start

        y_true = self._labels_to_windows(test_labels_raw, self.sequence_length)

        if len(y_true) != len(test_preds):
            raise ValueError(
                "DL label/prediction length mismatch: "
                f"labels={len(y_true)}, predictions={len(test_preds)}"
            )

        metrics = calculate_metrics(y_true, test_preds)

        return {
            "dataset": dataset_name,
            "fold": int(fold_idx),
            "model": model_name,
            "seed": int(seed),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "threshold": float(threshold),
            "threshold_percentile": float(threshold_percentile),
            "training_time_sec": float(training_time),
            "inference_time_sec": float(inference_time),
            "epochs_ran": int(len(train_losses)),
            "final_train_loss": float(train_losses[-1]) if train_losses else None,
            "final_val_loss": float(val_losses[-1]) if val_losses else None
        }

    def _merge_with_existing_results(
        self,
        new_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Merges newly generated DL results with existing result files.

        If the same dataset/model/seed/fold combination already exists,
        the newest row is kept. This prevents accidental overwriting while
        still allowing reruns to update previous failed or outdated entries.
        """
        csv_path = os.path.join(self.results_dir, "dl_experiment_results.csv")

        new_df = pd.DataFrame(new_results)

        if new_df.empty:
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
            return pd.DataFrame()

        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        dedup_keys = ["dataset", "model", "seed", "fold"]

        combined_df = combined_df.drop_duplicates(
            subset=dedup_keys,
            keep="last"
        )

        combined_df = combined_df.sort_values(
            by=["dataset", "model", "seed", "fold"]
        ).reset_index(drop=True)

        return combined_df

    def _save_results(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Saves detailed DL experiment results without overwriting old runs."""
        json_path = os.path.join(self.results_dir, "dl_experiment_results.json")
        csv_path = os.path.join(self.results_dir, "dl_experiment_results.csv")

        combined_df = self._merge_with_existing_results(all_results)

        combined_records = combined_df.to_dict(orient="records")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined_records, f, indent=4, ensure_ascii=False)

        combined_df.to_csv(csv_path, index=False)

        print(f"\nDetailed DL results saved to {csv_path}")

        return combined_df

    def _save_summary(self, results_df: Optional[pd.DataFrame] = None) -> None:
        """Saves mean/std summary grouped by dataset and model."""
        results_path = os.path.join(self.results_dir, "dl_experiment_results.csv")

        if results_df is None:
            if not os.path.exists(results_path):
                return
            results_df = pd.read_csv(results_path)

        if results_df.empty:
            return

        summary_df = (
            results_df.groupby(["dataset", "model"])
            .agg({
                "accuracy": ["mean", "std"],
                "precision": ["mean", "std"],
                "recall": ["mean", "std"],
                "f1": ["mean", "std"],
                "training_time_sec": ["mean", "std"],
                "inference_time_sec": ["mean", "std"]
            })
        )

        summary_df.columns = [
            "_".join(col).strip()
            for col in summary_df.columns.values
        ]

        summary_df = summary_df.reset_index()

        csv_path = os.path.join(self.results_dir, "dl_experiment_summary.csv")
        json_path = os.path.join(self.results_dir, "dl_experiment_summary.json")

        summary_df.to_csv(csv_path, index=False)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                summary_df.to_dict(orient="records"),
                f,
                indent=4,
                ensure_ascii=False
            )

        print(f"DL summary saved to {csv_path}")

    def run(
        self,
        datasets: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Runs DL experiments across datasets, models, and seeds."""
        if datasets is None:
            datasets = ["skab", "batadal"]

        if models is None:
            models = ["lstm", "cnn"]

        if seeds is None:
            seeds = self.seeds

        all_results: List[Dict[str, Any]] = []

        print("--- Starting Deep Learning Experiments ---")
        print(f"Device: {self.device}")

        for dataset_name in datasets:
            dataset_dirs = self._get_dataset_dirs(dataset_name)

            for model_name in models:
                for seed in seeds:
                    self.set_seed(seed)

                    print(
                        f"\n[DL] Dataset={dataset_name.upper()} | "
                        f"Model={model_name.upper()} | Seed={seed}"
                    )

                    fold_results = []

                    for fold_dir, fold_idx in dataset_dirs:
                        print(f"  Fold {fold_idx}")
                        result = self._evaluate_single_fold(
                            dataset_name=dataset_name,
                            fold_dir=fold_dir,
                            fold_idx=fold_idx,
                            model_name=model_name,
                            seed=seed
                        )
                        fold_results.append(result)
                        all_results.append(result)

                    mean_f1 = float(np.mean([r["f1"] for r in fold_results]))
                    print(f"  Mean F1: {mean_f1:.4f}")

        combined_results_df = self._save_results(all_results)
        self._save_summary(combined_results_df)

        return all_results


if __name__ == "__main__":
    runner = DeepLearningExperimentRunner("configs/config.yaml")
    runner.run()