"""
Training Pipeline
=================

Orchestrates the complete training workflow with dependency injection.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.config import Settings, get_settings
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.exceptions import DataValidationError, ModelTrainingError
from src.features.engineer import FeatureEngineer
from src.features.preprocessor import Preprocessor
from src.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Orchestrates the complete training pipeline.

    Supports dependency injection for all components, enabling:
    - Easy testing with mocks
    - Flexible component swapping
    - Configuration-driven behavior

    Steps:
    1. Load data
    2. Validate data
    3. Feature engineering
    4. Preprocessing
    5. Train model
    6. Save artifacts
    """

    def __init__(
        self,
        # Injected dependencies
        data_loader: DataLoader,
        data_validator: DataValidator,
        feature_engineer: FeatureEngineer,
        preprocessor: Preprocessor,
        model_trainer: ModelTrainer,
        # Configuration
        settings: Optional[Settings] = None,
        # Legacy parameters (deprecated, use settings)
        artifacts_dir: Optional[str] = None,
    ):
        """
        Initialize the TrainingPipeline with injected dependencies.

        Args:
            data_loader: DataLoader instance for loading data.
            data_validator: DataValidator instance for validation.
            feature_engineer: FeatureEngineer instance for feature creation.
            preprocessor: Preprocessor instance for data transformation.
            model_trainer: ModelTrainer instance for training.
            settings: Optional Settings instance for configuration.
            artifacts_dir: Deprecated. Use settings.artifacts.models_dir instead.
        """
        # Store injected dependencies
        self.data_loader = data_loader
        self.data_validator = data_validator
        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor
        self.trainer = model_trainer

        # Configuration
        self.settings = settings or get_settings()

        # Artifacts directory (with deprecation support)
        if artifacts_dir is not None:
            logger.warning(
                "artifacts_dir parameter is deprecated. "
                "Use settings.artifacts.models_dir instead."
            )
            self._artifacts_dir = Path(artifacts_dir)
        else:
            self._artifacts_dir = Path(self.settings.artifacts.models_dir)

        # Ensure artifacts directory exists
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "TrainingPipeline":
        """
        Factory method to create pipeline from settings (legacy support).

        This maintains backward compatibility while encouraging
        use of the Container for dependency injection.

        Args:
            settings: Optional Settings instance.

        Returns:
            Configured TrainingPipeline instance.
        """
        settings = settings or get_settings()

        return cls(
            data_loader=DataLoader(data_path=settings.data.raw_path),
            data_validator=DataValidator(),
            feature_engineer=FeatureEngineer(),
            preprocessor=Preprocessor(),
            model_trainer=ModelTrainer(
                experiment_name=settings.mlflow.experiment_name,
                tracking_uri=settings.mlflow.tracking_uri,
                random_state=settings.training.random_state,
            ),
            settings=settings,
        )

    @property
    def artifacts_dir(self) -> Path:
        """Return the artifacts directory path."""
        return self._artifacts_dir

    def run(
        self,
        run_name: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        test_size: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.

        Args:
            run_name: Optional name for the MLflow run.
            model_params: Optional XGBoost parameters (overrides settings).
            test_size: Test set proportion (overrides settings).

        Returns:
            Dictionary with training results and metrics.

        Raises:
            DataValidationError: If data validation fails.
            ModelTrainingError: If model training fails.
            PipelineError: For other pipeline errors.
        """
        # Use settings values as defaults
        test_size = test_size or self.settings.training.test_size

        # Merge model params with settings
        effective_params = self.settings.model.xgboost_params.copy()
        if model_params:
            effective_params.update(model_params)

        logger.info("=" * 50)
        logger.info("Starting Training Pipeline")
        logger.info(f"Environment: {self.settings.environment}")
        logger.info("=" * 50)

        try:
            # Step 1: Load Data
            logger.info("Step 1: Loading data...")
            df = self.data_loader.load_csv()
            logger.info(f"Loaded {len(df)} records")

            # Step 2: Validate Data
            logger.info("Step 2: Validating data...")
            validation_result = self.data_validator.validate(df)

            if not validation_result.is_valid:
                raise DataValidationError(
                    message="Data validation failed",
                    errors=validation_result.errors,
                )

            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(warning)

            logger.info(f"Data stats: {validation_result.stats}")

            # Step 3: Feature Engineering
            logger.info("Step 3: Applying feature engineering...")
            df = self.feature_engineer.fit_transform(df)

            # Step 4: Prepare features and target
            logger.info("Step 4: Preparing features and target...")
            target_col = self.settings.model.target_column
            y = df[target_col].values
            X_df = df.drop(columns=[target_col])

            # Step 5: Preprocess features
            logger.info("Step 5: Preprocessing features...")
            X = self.preprocessor.fit_transform(X_df)
            feature_names = self.preprocessor.get_feature_names_out()

            # Step 6: Split data
            logger.info("Step 6: Splitting data...")
            X_train, X_test, y_train, y_test = self.trainer.split_data(
                X, y, test_size=test_size
            )

            # Step 7: Train model
            logger.info("Step 7: Training model...")
            metrics = self.trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                model_params=effective_params,
                run_name=run_name,
                artifacts_dir=self._artifacts_dir,
            )

            # Step 8: Save artifacts
            logger.info("Step 8: Saving artifacts...")
            self._save_artifacts()

            logger.info("=" * 50)
            logger.info("Training Pipeline Complete!")
            logger.info(f"Final Metrics: {metrics}")
            logger.info("=" * 50)

            return {
                "metrics": metrics,
                "validation_stats": validation_result.stats,
                "feature_count": len(feature_names),
                "train_size": len(X_train),
                "test_size": len(X_test),
            }

        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise ModelTrainingError(
                message=f"Training pipeline failed: {str(e)}",
                original_error=e,
            )

    def _save_artifacts(self) -> None:
        """Save preprocessor and feature engineer for inference."""
        # Save preprocessor
        preprocessor_path = (
            self._artifacts_dir / self.settings.artifacts.preprocessor_filename
        )
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"Preprocessor saved to: {preprocessor_path}")

        # Save feature engineer
        engineer_path = (
            self._artifacts_dir / self.settings.artifacts.feature_engineer_filename
        )
        joblib.dump(self.feature_engineer, engineer_path)
        logger.info(f"Feature engineer saved to: {engineer_path}")


def main():
    """Main entry point for training pipeline."""
    import argparse

    from src.container import Container
    from src.utils.logging_config import setup_logging

    parser = argparse.ArgumentParser(description="Train hotel cancellation model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
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
        default=None,
        help="Test set proportion (overrides config)",
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["development", "staging", "production"],
        default=None,
        help="Environment override",
    )

    args = parser.parse_args()

    # Create container with configuration
    container = Container.from_settings(
        config_path=args.config,
    )

    # Setup logging from settings
    setup_logging(level=getattr(logging, container.settings.logging.level))

    # Create and run pipeline using container
    pipeline = container.training_pipeline()

    results = pipeline.run(
        run_name=args.run_name,
        test_size=args.test_size,
    )

    print(f"\nTraining Results: {results}")


if __name__ == "__main__":
    main()
