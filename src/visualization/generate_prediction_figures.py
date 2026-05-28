"""
Prediction Figure Generation Script
===================================
Loads trained DL checkpoints, runs inference on processed test sets, saves
prediction-level CSV files, and generates confusion matrix and precision-recall
curve figures.

This script does not retrain models.
"""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

from src.experiments.dl_experiments import DeepLearningExperimentRunner
from src.models.dl.data_loader import create_dataloader
from src.models.dl.evaluate import evaluate_model, detect_anomalies


RESULTS_DIR = "results"
FIGURES_DIR = "figures"
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


def save_current_figure(filename: str) -> None:
    """Saves the active matplotlib figure."""
    path = os.path.join(FIGURES_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def get_dataset_dirs(runner: DeepLearningExperimentRunner, dataset_name: str) -> List[Tuple[str, int]]:
    """Uses the runner's processed dataset directory logic."""
    return runner._get_dataset_dirs(dataset_name)


def load_test_data_and_labels(
    runner: DeepLearningExperimentRunner,
    fold_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads test PCA values and converts point-level labels into window-level labels."""
    test_pca = np.load(os.path.join(fold_dir, "test_pca.npy")).flatten()

    test_labels_raw = runner._load_labels(
        os.path.join(fold_dir, "test_labels.csv"),
        len(test_pca)
    )

    y_true = runner._labels_to_windows(
        test_labels_raw,
        runner.sequence_length
    )

    return test_pca, y_true


def load_checkpoint_model(
    runner: DeepLearningExperimentRunner,
    dataset_name: str,
    model_name: str,
    seed: int,
    fold_idx: int
) -> torch.nn.Module:
    """Builds the model architecture and loads the trained checkpoint."""
    checkpoint_path = os.path.join(
        runner.model_dir,
        f"best_{model_name}_{dataset_name}_seed{seed}_fold{fold_idx}.pt"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = runner._build_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=runner.device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(runner.device)
    model.eval()

    return model


def collect_predictions(
    dataset_name: str,
    model_name: str,
    seeds: List[int]
) -> pd.DataFrame:
    """
    Collects prediction-level results for selected dataset/model across seeds and folds.
    """
    runner = DeepLearningExperimentRunner("configs/config.yaml")
    dataset_dirs = get_dataset_dirs(runner, dataset_name)

    detailed_rows = []
    summary_df = pd.read_csv(os.path.join(RESULTS_DIR, "dl_experiment_results.csv"))

    for seed in seeds:
        runner.set_seed(seed)

        for fold_dir, fold_idx in dataset_dirs:
            print(f"[Inference] dataset={dataset_name}, model={model_name}, seed={seed}, fold={fold_idx}")

            model = load_checkpoint_model(
                runner=runner,
                dataset_name=dataset_name,
                model_name=model_name,
                seed=seed,
                fold_idx=fold_idx
            )

            test_pca, y_true = load_test_data_and_labels(runner, fold_dir)

            test_loader = create_dataloader(
                test_pca,
                runner.sequence_length,
                runner.batch_size,
                shuffle=False
            )

            test_scores = evaluate_model(
                model,
                test_loader,
                runner.device
            )

            match = summary_df[
                (summary_df["dataset"] == dataset_name)
                & (summary_df["model"] == model_name)
                & (summary_df["seed"] == seed)
                & (summary_df["fold"] == fold_idx)
            ]

            if match.empty:
                raise ValueError(
                    "Could not find threshold in dl_experiment_results.csv for "
                    f"{dataset_name}-{model_name}-seed{seed}-fold{fold_idx}"
                )

            threshold = float(match.iloc[0]["threshold"])
            y_pred = detect_anomalies(test_scores, threshold).astype(int)

            if len(y_true) != len(y_pred):
                raise ValueError(
                    f"Length mismatch for {dataset_name}-{model_name}-seed{seed}-fold{fold_idx}: "
                    f"y_true={len(y_true)}, y_pred={len(y_pred)}"
                )

            for window_index, (true_label, pred_label, score) in enumerate(
                zip(y_true, y_pred, test_scores)
            ):
                detailed_rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": int(seed),
                    "fold": int(fold_idx),
                    "window_index": int(window_index),
                    "y_true": int(true_label),
                    "y_pred": int(pred_label),
                    "score": float(score),
                    "threshold": float(threshold)
                })

    predictions_df = pd.DataFrame(detailed_rows)

    output_path = os.path.join(
        PREDICTIONS_DIR,
        f"{dataset_name}_{model_name}_predictions.csv"
    )

    predictions_df.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")

    return predictions_df


def plot_confusion_matrix_from_predictions(
    predictions_df: pd.DataFrame,
    dataset_name: str,
    model_name: str
) -> None:
    """Plots confusion matrix from prediction-level rows."""
    y_true = predictions_df["y_true"].values
    y_pred = predictions_df["y_pred"].values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {dataset_name.upper()} {model_name.upper()}")
    plt.colorbar()

    tick_labels = ["Normal", "Anomaly"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], tick_labels)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            plt.text(
                col_idx,
                row_idx,
                str(cm[row_idx, col_idx]),
                ha="center",
                va="center"
            )

    save_current_figure(
        f"confusion_matrix_{dataset_name}_{model_name}.png"
    )


def plot_precision_recall_curve_from_predictions(
    predictions_df: pd.DataFrame,
    dataset_name: str,
    model_name: str
) -> None:
    """Plots precision-recall curve from anomaly scores."""
    y_true = predictions_df["y_true"].values
    scores = predictions_df["score"].values

    if len(np.unique(y_true)) < 2:
        print(
            f"Skipping PR curve for {dataset_name}-{model_name}: "
            "only one class exists in y_true."
        )
        return

    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {dataset_name.upper()} {model_name.upper()}")
    plt.legend(loc="best")
    save_current_figure(
        f"pr_curve_{dataset_name}_{model_name}.png"
    )


def main() -> None:
    """Generates confusion matrix and PR curve figures for selected final models."""
    print("--- Generating prediction-based figures ---")

    seeds = [42, 123, 2026, 7, 999]

    selected_runs = [
        ("batadal", "cnn"),
        ("skab", "lstm")
    ]

    for dataset_name, model_name in selected_runs:
        predictions_df = collect_predictions(
            dataset_name=dataset_name,
            model_name=model_name,
            seeds=seeds
        )

        plot_confusion_matrix_from_predictions(
            predictions_df,
            dataset_name,
            model_name
        )

        plot_precision_recall_curve_from_predictions(
            predictions_df,
            dataset_name,
            model_name
        )

    print("--- Prediction-based figure generation completed ---")


if __name__ == "__main__":
    main()