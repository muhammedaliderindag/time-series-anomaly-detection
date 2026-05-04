"""
Experiment Logger Module
=========================
Provides robust logging for ML experiments including:
- Standard output logging (info, warning, error) to .log files
- Structured experiment metrics saving (F1, Accuracy, Precision, Recall)
- Execution time tracking (training time, inference time)

All paths are dynamically resolved from the configuration parser.
"""

import csv
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """Comprehensive logger for ML experiment tracking.

    Handles three concerns:
    1. Standard Python logging to console + .log file
    2. Structured metrics persistence (per fold/seed) to JSON and CSV
    3. Wall-clock timing for training and inference phases
    """

    def __init__(self, config: dict, experiment_name: Optional[str] = None) -> None:
        """Initialize the experiment logger.

        Args:
            config: Full configuration dictionary (from ConfigParser.config).
            experiment_name: Optional name for this experiment run.
                             Defaults to a timestamp-based name.
        """
        # Resolve log directory from config
        self._log_dir: str = config.get("paths", {}).get("log_dir", "./logs")
        os.makedirs(self._log_dir, exist_ok=True)

        # Experiment identity
        self._timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._experiment_name: str = experiment_name or f"experiment_{self._timestamp}"

        # --- Standard Python Logger ---
        self._logger: logging.Logger = logging.getLogger(self._experiment_name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        # File handler
        log_file = os.path.join(self._log_dir, f"{self._experiment_name}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

        # --- Metrics Storage ---
        self._metrics: List[Dict[str, Any]] = []

        # --- Timing ---
        self._timers: Dict[str, float] = {}
        self._recorded_times: Dict[str, float] = {}

        self.info(f"Logger initialized for experiment: {self._experiment_name}")
        self.info(f"Log directory: {os.path.abspath(self._log_dir)}")

    # ------------------------------------------------------------------ #
    #  Standard Logging Methods
    # ------------------------------------------------------------------ #

    def info(self, message: str) -> None:
        """Log an informational message."""
        self._logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self._logger.error(message)

    def debug(self, message: str) -> None:
        """Log a debug-level message."""
        self._logger.debug(message)

    # ------------------------------------------------------------------ #
    #  Timing Methods
    # ------------------------------------------------------------------ #

    def start_timer(self, name: str) -> None:
        """Start a named timer.

        Args:
            name: Timer identifier (e.g., 'training_time', 'inference_time').
        """
        self._timers[name] = time.time()
        self.info(f"Timer '{name}' started.")

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed seconds.

        Args:
            name: Timer identifier that was previously started.

        Returns:
            Elapsed time in seconds.

        Raises:
            ValueError: If the timer was never started.
        """
        if name not in self._timers:
            raise ValueError(f"Timer '{name}' was never started.")

        elapsed = time.time() - self._timers.pop(name)
        self._recorded_times[name] = elapsed
        self.info(f"Timer '{name}' stopped. Elapsed: {elapsed:.4f}s")
        return elapsed

    def get_elapsed_time(self, name: str) -> Optional[float]:
        """Retrieve the recorded elapsed time for a timer.

        Args:
            name: Timer identifier.

        Returns:
            Elapsed seconds, or None if not recorded.
        """
        return self._recorded_times.get(name)

    # ------------------------------------------------------------------ #
    #  Metrics Tracking
    # ------------------------------------------------------------------ #

    def log_metrics(
        self,
        seed: int,
        fold: Optional[int] = None,
        **metrics: float,
    ) -> None:
        """Record experiment metrics for a given seed/fold.

        Args:
            seed: Random seed used for this run.
            fold: Optional cross-validation fold index.
            **metrics: Keyword arguments for metric values,
                       e.g., f1_score=0.95, accuracy=0.93, precision=0.91, recall=0.97.
        """
        entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "experiment": self._experiment_name,
            "seed": seed,
            "fold": fold,
        }
        entry.update(metrics)

        # Append recorded times
        for timer_name, elapsed in self._recorded_times.items():
            entry[f"{timer_name}_seconds"] = round(elapsed, 4)

        self._metrics.append(entry)

        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.info(f"Metrics [seed={seed}, fold={fold}]: {metrics_str}")

    def save_metrics(self, fmt: str = "both") -> None:
        """Persist collected metrics to disk.

        Args:
            fmt: Output format — 'json', 'csv', or 'both'.
        """
        if not self._metrics:
            self.warning("No metrics to save.")
            return

        base_path = os.path.join(self._log_dir, f"{self._experiment_name}_metrics")

        if fmt in ("json", "both"):
            json_path = f"{base_path}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self._metrics, f, indent=2, ensure_ascii=False)
            self.info(f"Metrics saved to {json_path}")

        if fmt in ("csv", "both"):
            csv_path = f"{base_path}.csv"
            if self._metrics:
                fieldnames = list(self._metrics[0].keys())
                # Collect all unique keys across entries
                for entry in self._metrics:
                    for key in entry:
                        if key not in fieldnames:
                            fieldnames.append(key)

                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self._metrics)
                self.info(f"Metrics saved to {csv_path}")

    # ------------------------------------------------------------------ #
    #  Summary
    # ------------------------------------------------------------------ #

    def summary(self) -> None:
        """Print a summary of all recorded metrics and times."""
        self.info("=" * 60)
        self.info("EXPERIMENT SUMMARY")
        self.info("=" * 60)
        self.info(f"Experiment: {self._experiment_name}")
        self.info(f"Total metric entries: {len(self._metrics)}")

        if self._recorded_times:
            self.info("Recorded times:")
            for name, elapsed in self._recorded_times.items():
                self.info(f"  {name}: {elapsed:.4f}s")

        if self._metrics:
            # Compute averages for numeric metrics
            numeric_keys = [
                k
                for k in self._metrics[0]
                if k not in ("timestamp", "experiment", "seed", "fold")
                and isinstance(self._metrics[0].get(k), (int, float))
            ]
            if numeric_keys:
                self.info("Average metrics across all entries:")
                for key in numeric_keys:
                    values = [
                        m[key] for m in self._metrics if key in m and m[key] is not None
                    ]
                    if values:
                        avg = sum(values) / len(values)
                        self.info(f"  {key}: {avg:.4f}")

        self.info("=" * 60)

    def __repr__(self) -> str:
        return (
            f"ExperimentLogger(name='{self._experiment_name}', "
            f"metrics_count={len(self._metrics)})"
        )
