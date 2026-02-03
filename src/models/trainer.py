"""
Model Training Module
=====================

Handles model training with MLflow integration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles model training with MLflow tracking.

    Responsible for:
    - Splitting data
    - Training XGBoost model
    - Evaluating performance
    - Logging to MLflow
    """

    def __init__(
        self,
        experiment_name: str = "hotel_cancellation_prediction",
        tracking_uri: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initialize the ModelTrainer.

        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: MLflow tracking server URI. None for local file store.
            random_state: Random seed for reproducibility.
        """
        self.experiment_name = experiment_name
        self.random_state = random_state
        self._model: Optional[XGBClassifier] = None
        self._feature_names: Optional[list] = None

        # Set up MLflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow experiment set to: {experiment_name}")

    @property
    def model(self) -> Optional[XGBClassifier]:
        """Return the trained model."""
        return self._model

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets.

        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of data for testing.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,  # Maintain class distribution
        )

        logger.info(
            f"Data split: Train={len(X_train)}, Test={len(X_test)} "
            f"(test_size={test_size})"
        )

        return X_train, X_test, y_train, y_test

    def _calculate_scale_pos_weight(self, y: np.ndarray) -> float:
        """
        Calculate scale_pos_weight for imbalanced classes.

        Args:
            y: Target vector.

        Returns:
            Ratio of negative to positive samples.
        """
        n_negative = np.sum(y == 0)
        n_positive = np.sum(y == 1)

        if n_positive == 0:
            return 1.0

        weight = n_negative / n_positive
        logger.info(f"Class imbalance - scale_pos_weight: {weight:.2f}")

        return weight

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[list] = None,
        model_params: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        artifacts_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Train the model and log to MLflow.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            y_test: Test target.
            feature_names: List of feature names for importance plot.
            model_params: Optional XGBoost parameters to override defaults.
            run_name: Optional name for the MLflow run.
            artifacts_dir: Directory to save artifacts locally.

        Returns:
            Dictionary of evaluation metrics.
        """
        self._feature_names = feature_names

        # Calculate class weight for imbalanced data
        scale_pos_weight = self._calculate_scale_pos_weight(y_train)

        # Base parameters (only runtime-computed values)
        base_params = {
            "scale_pos_weight": scale_pos_weight,
            "random_state": self.random_state,
            "use_label_encoder": False,
        }

        # Merge: model_params from settings take priority, then base_params
        # This ensures settings/YAML/env vars control the hyperparameters
        effective_params = {**base_params, **(model_params or {})}

        # Start MLflow run
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"Started MLflow run: {run.info.run_id}")

            # Enable autologging for XGBoost
            mlflow.xgboost.autolog(log_models=False)  # We'll log model manually

            # Log parameters
            mlflow.log_params(effective_params)

            # Train model
            self._model = XGBClassifier(**effective_params)
            self._model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            # Evaluate
            metrics = self._evaluate(X_test, y_test)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Generate and log artifacts
            if artifacts_dir:
                artifacts_dir = Path(artifacts_dir)
                artifacts_dir.mkdir(parents=True, exist_ok=True)

            self._log_confusion_matrix(y_test, X_test, artifacts_dir)
            self._log_feature_importance(artifacts_dir)

            # Log model
            mlflow.xgboost.log_model(
                self._model,
                artifact_path="model",
                registered_model_name="hotel_cancellation_model",
            )

            logger.info(f"Training complete. Metrics: {metrics}")

            return metrics

    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features.
            y_test: Test target.

        Returns:
            Dictionary of metrics.
        """
        y_pred = self._model.predict(X_test)
        y_pred_proba = self._model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        # Log classification report
        logger.info(
            f"\nClassification Report:\n{classification_report(y_test, y_pred)}"
        )

        return metrics

    def _log_confusion_matrix(
        self,
        y_test: np.ndarray,
        X_test: np.ndarray,
        artifacts_dir: Optional[Path] = None,
    ) -> None:
        """
        Generate and log confusion matrix plot.

        Args:
            y_test: True labels.
            X_test: Test features.
            artifacts_dir: Directory to save artifact.
        """
        y_pred = self._model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Canceled", "Canceled"],
            yticklabels=["Not Canceled", "Canceled"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        # Save locally and log to MLflow
        if artifacts_dir:
            cm_path = artifacts_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(str(cm_path))
        else:
            cm_path = "/tmp/confusion_matrix.png"  # nosec: B108
            plt.savefig(cm_path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(cm_path)

        plt.close()
        logger.info("Confusion matrix logged to MLflow")

    def _log_feature_importance(self, artifacts_dir: Optional[Path] = None) -> None:
        """
        Generate and log feature importance plot.

        Args:
            artifacts_dir: Directory to save artifact.
        """
        importance = self._model.feature_importances_

        # Use feature names if available
        if self._feature_names and len(self._feature_names) == len(importance):
            feature_names = self._feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        # Create DataFrame for plotting
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=True)

        # Plot top 20 features
        plt.figure(figsize=(10, 8))
        top_n = min(20, len(importance_df))
        plt.barh(
            importance_df["feature"].tail(top_n),
            importance_df["importance"].tail(top_n),
        )
        plt.xlabel("Feature Importance")
        plt.title("Top Feature Importances (XGBoost)")
        plt.tight_layout()

        # Save and log
        if artifacts_dir:
            fi_path = artifacts_dir / "feature_importance.png"
            plt.savefig(fi_path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(str(fi_path))
        else:
            fi_path = "/tmp/feature_importance.png"  # nosec: B108
            plt.savefig(fi_path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(fi_path)

        plt.close()
        logger.info("Feature importance plot logged to MLflow")
