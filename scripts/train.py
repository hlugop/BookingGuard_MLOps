#!/usr/bin/env python
"""
Training Script
===============

Convenience script to run the training pipeline.

Usage:
    python scripts/train.py
    python scripts/train.py --run-name "experiment_1"
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.training import TrainingPipeline
from src.utils.logging_config import setup_logging


def main():
    """Run the training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Train the hotel cancellation model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/Hotel Reservations.csv",
        help="Path to training data",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/models",
        help="Directory to save artifacts",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="hotel_cancellation_prediction",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.INFO)

    # Run pipeline
    pipeline = TrainingPipeline(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        experiment_name=args.experiment_name,
    )

    results = pipeline.run(
        run_name=args.run_name,
        test_size=args.test_size,
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"\nMetrics:")
    for metric, value in results["metrics"].items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nFeatures: {results['feature_count']}")
    print(f"Train size: {results['train_size']}")
    print(f"Test size: {results['test_size']}")
    print("\nView experiments at: http://localhost:5000 (run 'mlflow ui')")


if __name__ == "__main__":
    main()
