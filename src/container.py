"""
Dependency Injection Container
==============================

Provides a centralized container for managing application dependencies.
Enables loose coupling, easier testing, and flexible component swapping.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.features.store import FeatureStore

import joblib

from src.config import Settings, get_settings
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.exceptions import ArtifactNotFoundError
from src.features.engineer import FeatureEngineer
from src.features.preprocessor import Preprocessor
from src.models.predictor import ModelPredictor
from src.models.trainer import ModelTrainer


@dataclass
class Container:
    """
    Dependency Injection Container.

    Manages the lifecycle and dependencies of all application components.
    Supports lazy initialization and easy mocking for tests.

    Usage:
        # Production
        container = Container.from_settings()

        # Testing with mocks
        container = Container(
            data_loader=mock_loader,
            validator=mock_validator,
        )

        # Access components
        pipeline = container.training_pipeline()
    """

    # Configuration
    settings: Settings = field(default_factory=get_settings)

    # Core components (optional for DI)
    _data_loader: Optional[DataLoader] = field(default=None, repr=False)
    _data_validator: Optional[DataValidator] = field(default=None, repr=False)
    _feature_engineer: Optional[FeatureEngineer] = field(default=None, repr=False)
    _preprocessor: Optional[Preprocessor] = field(default=None, repr=False)
    _model_trainer: Optional[ModelTrainer] = field(default=None, repr=False)
    _model_predictor: Optional[ModelPredictor] = field(default=None, repr=False)
    _feature_store: Optional["FeatureStore"] = field(default=None, repr=False)

    # Lazy loading flags
    _artifacts_loaded: bool = field(default=False, repr=False)

    @classmethod
    def from_settings(
        cls,
        settings: Optional[Settings] = None,
        config_path: Optional[str] = None,
    ) -> "Container":
        """
        Create container from settings.

        Args:
            settings: Optional Settings instance.
            config_path: Optional path to YAML config.

        Returns:
            Configured Container instance.
        """
        if settings is None:
            settings = get_settings(config_path=config_path)
        return cls(settings=settings)

    @classmethod
    def for_testing(
        cls,
        data_loader: Optional[DataLoader] = None,
        validator: Optional[DataValidator] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
        preprocessor: Optional[Preprocessor] = None,
        trainer: Optional[ModelTrainer] = None,
        predictor: Optional[ModelPredictor] = None,
    ) -> "Container":
        """
        Create container for testing with optional mocked components.

        Args:
            data_loader: Optional mock DataLoader.
            validator: Optional mock DataValidator.
            feature_engineer: Optional mock FeatureEngineer.
            preprocessor: Optional mock Preprocessor.
            trainer: Optional mock ModelTrainer.
            predictor: Optional mock ModelPredictor.

        Returns:
            Container configured for testing.
        """
        from src.config.settings import clear_settings_cache

        # Use testing environment
        clear_settings_cache()
        settings = get_settings(env="testing")

        return cls(
            settings=settings,
            _data_loader=data_loader,
            _data_validator=validator,
            _feature_engineer=feature_engineer,
            _preprocessor=preprocessor,
            _model_trainer=trainer,
            _model_predictor=predictor,
        )

    # =========================================================================
    # Component Providers (Lazy Initialization)
    # =========================================================================

    def data_loader(self, path: Optional[str] = None) -> DataLoader:
        """
        Get or create DataLoader instance.

        Args:
            path: Optional data path override.

        Returns:
            DataLoader instance.
        """
        if self._data_loader is not None:
            return self._data_loader

        data_path = path or self.settings.data.raw_path
        return DataLoader(data_path=data_path)

    def data_validator(self) -> DataValidator:
        """
        Get or create DataValidator instance.

        Returns:
            DataValidator instance.
        """
        if self._data_validator is not None:
            return self._data_validator

        return DataValidator()

    def feature_engineer(self) -> FeatureEngineer:
        """
        Get or create FeatureEngineer instance.

        Returns:
            FeatureEngineer instance.
        """
        if self._feature_engineer is not None:
            return self._feature_engineer

        return FeatureEngineer()

    def preprocessor(self) -> Preprocessor:
        """
        Get or create Preprocessor instance.

        Returns:
            Preprocessor instance.
        """
        if self._preprocessor is not None:
            return self._preprocessor

        return Preprocessor()

    def model_trainer(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ) -> ModelTrainer:
        """
        Get or create ModelTrainer instance.

        Args:
            experiment_name: Optional MLflow experiment name.
            tracking_uri: Optional MLflow tracking URI.

        Returns:
            ModelTrainer instance.
        """
        if self._model_trainer is not None:
            return self._model_trainer

        return ModelTrainer(
            experiment_name=experiment_name or self.settings.mlflow.experiment_name,
            tracking_uri=tracking_uri or self.settings.mlflow.tracking_uri,
            random_state=self.settings.training.random_state,
        )

    def model_predictor(
        self,
        model_uri: Optional[str] = None,
    ) -> ModelPredictor:
        """
        Get or create ModelPredictor instance.

        Args:
            model_uri: Optional model URI override.

        Returns:
            ModelPredictor instance.
        """
        if self._model_predictor is not None:
            return self._model_predictor

        uri = model_uri or self.settings.get_model_uri()

        return ModelPredictor(
            model_uri=uri,
            preprocessor=self.preprocessor(),
            feature_engineer=self.feature_engineer(),
        )

    def feature_store(self) -> "FeatureStore":
        """
        Get or create FeatureStore instance.

        Returns:
            FeatureStore instance.
        """
        if self._feature_store is not None:
            return self._feature_store

        from src.features.store import FeatureStore

        return FeatureStore.from_settings(self.settings)

    # =========================================================================
    # Artifact Management
    # =========================================================================

    def load_artifacts(self) -> None:
        """
        Load preprocessor and feature engineer from saved artifacts.

        Raises:
            ArtifactNotFoundError: If artifacts are not found.
        """
        if self._artifacts_loaded:
            return

        artifacts_dir = Path(self.settings.artifacts.models_dir)

        # Load preprocessor
        preprocessor_path = (
            artifacts_dir / self.settings.artifacts.preprocessor_filename
        )
        if not preprocessor_path.exists():
            raise ArtifactNotFoundError(
                artifact_name="preprocessor",
                artifact_path=str(preprocessor_path),
            )
        self._preprocessor = joblib.load(preprocessor_path)

        # Load feature engineer
        engineer_path = (
            artifacts_dir / self.settings.artifacts.feature_engineer_filename
        )
        if not engineer_path.exists():
            raise ArtifactNotFoundError(
                artifact_name="feature_engineer",
                artifact_path=str(engineer_path),
            )
        self._feature_engineer = joblib.load(engineer_path)

        self._artifacts_loaded = True

    def save_artifacts(
        self,
        preprocessor: Preprocessor,
        feature_engineer: FeatureEngineer,
    ) -> None:
        """
        Save preprocessor and feature engineer artifacts.

        Args:
            preprocessor: Fitted Preprocessor to save.
            feature_engineer: Fitted FeatureEngineer to save.
        """
        artifacts_dir = Path(self.settings.artifacts.models_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save preprocessor
        preprocessor_path = (
            artifacts_dir / self.settings.artifacts.preprocessor_filename
        )
        joblib.dump(preprocessor, preprocessor_path)

        # Save feature engineer
        engineer_path = (
            artifacts_dir / self.settings.artifacts.feature_engineer_filename
        )
        joblib.dump(feature_engineer, engineer_path)

        # Update internal references
        self._preprocessor = preprocessor
        self._feature_engineer = feature_engineer
        self._artifacts_loaded = True

    # =========================================================================
    # Pipeline Factories
    # =========================================================================

    def training_pipeline(self):
        """
        Create a configured TrainingPipeline instance.

        Returns:
            TrainingPipeline with injected dependencies.
        """
        from src.pipelines.training import TrainingPipeline

        return TrainingPipeline(
            data_loader=self.data_loader(),
            data_validator=self.data_validator(),
            feature_engineer=self.feature_engineer(),
            preprocessor=self.preprocessor(),
            model_trainer=self.model_trainer(),
            settings=self.settings,
        )

    def inference_pipeline(self, load_artifacts: bool = True):
        """
        Create a configured InferencePipeline instance.

        Args:
            load_artifacts: Whether to load saved artifacts.
                If False, the pipeline will load artifacts when load() is called.

        Returns:
            InferencePipeline with injected dependencies.
        """
        from src.pipelines.inference import InferencePipeline

        if load_artifacts:
            self.load_artifacts()
            # When artifacts are loaded, pass them to the predictor
            return InferencePipeline(
                predictor=self.model_predictor(),
                feature_engineer=self.feature_engineer(),
                preprocessor=self.preprocessor(),
                settings=self.settings,
            )
        else:
            # When load_artifacts=False, let InferencePipeline handle loading
            # Don't pass predictor - it will be created during load()
            return InferencePipeline(
                predictor=None,
                feature_engineer=None,
                preprocessor=None,
                settings=self.settings,
            )


# Global container instance (optional, for simple use cases)
_container: Optional[Container] = None


def get_container(
    config_path: Optional[str] = None,
    reset: bool = False,
) -> Container:
    """
    Get or create the global container instance.

    Args:
        config_path: Optional path to config file.
        reset: If True, recreate the container.

    Returns:
        Container instance.
    """
    global _container

    if _container is None or reset:
        _container = Container.from_settings(config_path=config_path)

    return _container


def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _container
    _container = None
