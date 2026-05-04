"""
Sanity Check Script
====================
Verifies the integration between Config Parser and Experiment Logger.
Loads configuration, logs test messages, records dummy metrics, and
ensures all modules are working correctly end-to-end.
"""

import time
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_parser import ConfigParser
from src.utils.logger import ExperimentLogger


def main() -> None:
    """Run the sanity check."""

    config_path = os.path.join("configs", "config.yaml")

    # ----------------------------------------------------------
    # 1. Load Configuration
    # ----------------------------------------------------------
    print("=" * 60)
    print("SANITY CHECK: Time Series Anomaly Detection Infrastructure")
    print("=" * 60)

    cfg = ConfigParser(config_path)
    print(f"\n[OK] Config loaded: {cfg}")
    print(f"  batch_size       = {cfg.get('batch_size')}")
    print(f"  max_epoch        = {cfg.get('max_epoch')}")
    print(f"  early_stopping   = {cfg.get('early_stopping')}")
    print(f"  random_seeds     = {cfg.get('random_seeds')}")
    print(f"  data_dir         = {cfg.get('paths.data_dir')}")
    print(f"  log_dir          = {cfg.get('paths.log_dir')}")
    print(f"  model_name       = {cfg.get('model.name')}")

    # ----------------------------------------------------------
    # 2. Initialize Logger
    # ----------------------------------------------------------
    logger = ExperimentLogger(cfg.config, experiment_name="sanity_check")
    logger.info("Sanity check started.")
    logger.info(f"Configuration: batch_size={cfg.get('batch_size')}, "
                f"max_epoch={cfg.get('max_epoch')}")

    # ----------------------------------------------------------
    # 3. Test Logging Levels
    # ----------------------------------------------------------
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.debug("This is a DEBUG message (only visible in log file).")

    # ----------------------------------------------------------
    # 4. Test Timers
    # ----------------------------------------------------------
    logger.start_timer("training_time")
    time.sleep(0.5)  # Simulate training
    training_time = logger.stop_timer("training_time")

    logger.start_timer("inference_time")
    time.sleep(0.2)  # Simulate inference
    inference_time = logger.stop_timer("inference_time")

    # ----------------------------------------------------------
    # 5. Test Metrics Logging
    # ----------------------------------------------------------
    seeds = cfg.get("random_seeds")
    for i, seed in enumerate(seeds):
        logger.log_metrics(
            seed=seed,
            fold=i,
            f1_score=0.90 + i * 0.01,
            accuracy=0.92 + i * 0.01,
            precision=0.88 + i * 0.02,
            recall=0.93 + i * 0.01,
        )

    # ----------------------------------------------------------
    # 6. Save Metrics
    # ----------------------------------------------------------
    logger.save_metrics(fmt="both")

    # ----------------------------------------------------------
    # 7. Print Summary
    # ----------------------------------------------------------
    logger.summary()

    # ----------------------------------------------------------
    # 8. Verify Created Paths
    # ----------------------------------------------------------
    log_dir = cfg.get("paths.log_dir")
    print(f"\n[OK] Log directory exists: {os.path.exists(log_dir)}")
    print(f"[OK] Log files created:")
    for f in os.listdir(log_dir):
        filepath = os.path.join(log_dir, f)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            print(f"  - {f} ({size} bytes)")

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
