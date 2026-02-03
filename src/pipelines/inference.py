"""
Inference Pipeline
==================

Orchestrates the prediction workflow with dependency injection.
"""

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd

from src.config import Settings, get_settings
from src.exceptions import (
    ArtifactNotFoundError,
    ModelLoadError,
    PipelineNotLoadedError,
    PredictionError,
)
from src.features.engineer import FeatureEngineer
from src.features.preprocessor import Preprocessor
from src.models.predictor import ModelPredictor

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Orchestrates the inference/prediction workflow.

    Supports dependency injection for all components, enabling:
    - Easy testing with mocks
    - Flexible component swapping
    - Configuration-driven behavior

    Steps:
    1. Load model and preprocessor
    2. Validate input
    3. Transform input
    4. Make predictions
    """

    def __init__(
        self,
        # Injected dependencies
        predictor: Optional[ModelPredictor] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
        preprocessor: Optional[Preprocessor] = None,
        # Configuration
        settings: Optional[Settings] = None,
        # Legacy parameters (deprecated)
        model_uri: Optional[str] = None,
        artifacts_dir: Optional[str] = None,
    ):
        """
        Initialize the InferencePipeline with injected dependencies.

        Args:
            predictor: ModelPredictor instance for predictions.
            feature_engineer: FeatureEngineer instance for feature creation.
            preprocessor: Preprocessor instance for data transformation.
            settings: Optional Settings instance for configuration.
            model_uri: Deprecated. Use settings.api.model_uri instead.
            artifacts_dir: Deprecated. Use settings.artifacts.models_dir instead.
        """
        # Configuration
        self.settings = settings or get_settings()

        # Handle deprecated parameters
        if model_uri is not None:
            logger.warning(
                "model_uri parameter is deprecated. "
                "Use settings.api.model_uri instead."
            )
            self._model_uri = model_uri
        else:
            self._model_uri = self.settings.get_model_uri()

        if artifacts_dir is not None:
            logger.warning(
                "artifacts_dir parameter is deprecated. "
                "Use settings.artifacts.models_dir instead."
            )
            self._artifacts_dir = Path(artifacts_dir)
        else:
            self._artifacts_dir = Path(self.settings.artifacts.models_dir)

        # Store injected dependencies (can be None initially)
        self._predictor = predictor
        self._feature_engineer = feature_engineer
        self._preprocessor = preprocessor

        # Track loading state
        self._is_loaded: bool = False

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "InferencePipeline":
        """
        Factory method to create pipeline from settings.

        Args:
            settings: Optional Settings instance.

        Returns:
            Configured InferencePipeline instance.
        """
        settings = settings or get_settings()
        return cls(settings=settings)

    @property
    def is_loaded(self) -> bool:
        """Check if the pipeline is ready for inference."""
        return self._is_loaded

    @property
    def model_uri(self) -> str:
        """Return the model URI."""
        return self._model_uri

    def load(self, model_uri: Optional[str] = None) -> "InferencePipeline":
        """
        Load model and preprocessor for inference.

        Args:
            model_uri: Optional model URI override.

        Returns:
            Self for method chaining.

        Raises:
            ArtifactNotFoundError: If preprocessor/feature_engineer not found.
            ModelLoadError: If model loading fails.
        """
        uri = model_uri or self._model_uri

        # Load preprocessor if not injected
        if self._preprocessor is None:
            self._preprocessor = self._load_artifact(
                self.settings.artifacts.preprocessor_filename,
                "preprocessor",
            )
            logger.info(f"Loaded preprocessor from: {self._artifacts_dir}")

        # Load feature engineer if not injected
        if self._feature_engineer is None:
            self._feature_engineer = self._load_artifact(
                self.settings.artifacts.feature_engineer_filename,
                "feature_engineer",
            )
            logger.info(f"Loaded feature engineer from: {self._artifacts_dir}")

        # Create predictor if not injected
        if self._predictor is None:
            self._predictor = ModelPredictor(
                model_uri=uri,
                preprocessor=self._preprocessor,
                feature_engineer=self._feature_engineer,
            )

        # Ensure predictor has preprocessor and feature engineer
        if self._predictor._preprocessor is None:
            self._predictor.set_preprocessor(self._preprocessor)

        # Load model
        try:
            self._predictor.load_model(uri)
            logger.info(f"Loaded model from: {uri}")
        except Exception as e:
            raise ModelLoadError(
                message=f"Failed to load model: {str(e)}",
                model_uri=uri,
                original_error=e,
            )

        self._is_loaded = True
        return self

    def _load_artifact(self, filename: str, artifact_name: str):
        """
        Load an artifact from the artifacts directory.

        Args:
            filename: Name of the artifact file.
            artifact_name: Human-readable name for error messages.

        Returns:
            Loaded artifact.

        Raises:
            ArtifactNotFoundError: If artifact file not found.
        """
        artifact_path = self._artifacts_dir / filename

        if not artifact_path.exists():
            raise ArtifactNotFoundError(
                artifact_name=artifact_name,
                artifact_path=str(artifact_path),
            )

        return joblib.load(artifact_path)

    def _ensure_loaded(self) -> None:
        """
        Ensure the pipeline is loaded before making predictions.

        Raises:
            PipelineNotLoadedError: If pipeline not loaded.
        """
        if not self._is_loaded:
            raise PipelineNotLoadedError(
                message="Pipeline not loaded. Call load() first."
            )

    def predict_single(self, data: dict) -> dict:
        """
        Make prediction for a single record.

        Args:
            data: Single record as dictionary.

        Returns:
            Prediction result with prediction, probability, and label.

        Raises:
            PipelineNotLoadedError: If pipeline not loaded.
            PredictionError: If prediction fails.
        """
        self._ensure_loaded()

        try:
            return self._predictor.predict_single(data)
        except Exception as e:
            raise PredictionError(
                message=f"Single prediction failed: {str(e)}",
                original_error=e,
            )

    def predict_batch(self, data: List[dict]) -> List[dict]:
        """
        Make predictions for multiple records.

        Args:
            data: List of records as dictionaries.

        Returns:
            List of prediction results.

        Raises:
            PipelineNotLoadedError: If pipeline not loaded.
            PredictionError: If prediction fails.
        """
        self._ensure_loaded()

        try:
            return self._predictor.predict_batch(data)
        except Exception as e:
            raise PredictionError(
                message=f"Batch prediction failed: {str(e)}",
                input_shape=(len(data),),
                original_error=e,
            )

    def predict_dataframe(
        self, df: pd.DataFrame, return_proba: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for a DataFrame.

        Args:
            df: Input DataFrame.
            return_proba: If True, include probabilities.

        Returns:
            DataFrame with predictions appended.

        Raises:
            PipelineNotLoadedError: If pipeline not loaded.
            PredictionError: If prediction fails.
        """
        self._ensure_loaded()

        try:
            result = self._predictor.predict(df, return_proba=return_proba)

            if isinstance(result, dict):
                df_result = df.copy()
                df_result["prediction"] = result["predictions"]
                df_result["probability"] = result["probabilities"]
                df_result["label"] = result["labels"]
                return df_result

            df_result = df.copy()
            df_result["prediction"] = result
            return df_result
        except Exception as e:
            raise PredictionError(
                message=f"DataFrame prediction failed: {str(e)}",
                input_shape=df.shape,
                original_error=e,
            )


def main():
    """Main entry point for inference pipeline."""
    import argparse

    from src.container import Container
    from src.utils.logging_config import setup_logging

    parser = argparse.ArgumentParser(description="Run inference pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        default=None,
        help="MLflow model URI (overrides config)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input CSV file for predictions",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="predictions.csv",
        help="Output CSV file for predictions",
    )

    args = parser.parse_args()

    # Create container with configuration
    container = Container.from_settings(config_path=args.config)

    # Setup logging from settings
    setup_logging(level=getattr(logging, container.settings.logging.level))

    # Load pipeline
    pipeline = container.inference_pipeline(load_artifacts=False)
    pipeline.load(model_uri=args.model_uri)

    # Load input data
    df = pd.read_csv(args.input_file)
    logger.info(f"Loaded {len(df)} records from {args.input_file}")

    # Make predictions
    results = pipeline.predict_dataframe(df)

    # Save results
    results.to_csv(args.output_file, index=False)
    logger.info(f"Saved predictions to {args.output_file}")

    print(f"\nPredictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
