"""
Unit Tests for Dependency Injection Container
=============================================

Tests for the Container class and dependency management.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings, clear_settings_cache
from src.container import Container, get_container, reset_container
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.features.engineer import FeatureEngineer
from src.features.preprocessor import Preprocessor
from src.models.trainer import ModelTrainer


class TestContainer:
    """Tests for Container class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        clear_settings_cache()
        reset_container()
        yield
        clear_settings_cache()
        reset_container()

    def test_from_settings_creates_container(self):
        """Test that from_settings creates a valid container."""
        container = Container.from_settings()
        assert container is not None
        assert container.settings is not None

    def test_from_settings_with_custom_settings(self):
        """Test from_settings with custom settings."""
        settings = Settings(environment="testing")
        container = Container.from_settings(settings=settings)
        assert container.settings.environment == "testing"

    def test_data_loader_lazy_creation(self):
        """Test that data_loader is created lazily."""
        container = Container.from_settings()
        loader = container.data_loader()
        assert isinstance(loader, DataLoader)

    def test_data_loader_with_injected_dependency(self):
        """Test that injected data_loader is used."""
        mock_loader = MagicMock(spec=DataLoader)
        container = Container.for_testing(data_loader=mock_loader)
        loader = container.data_loader()
        assert loader is mock_loader

    def test_data_validator_lazy_creation(self):
        """Test that data_validator is created lazily."""
        container = Container.from_settings()
        validator = container.data_validator()
        assert isinstance(validator, DataValidator)

    def test_data_validator_with_injected_dependency(self):
        """Test that injected data_validator is used."""
        mock_validator = MagicMock(spec=DataValidator)
        container = Container.for_testing(validator=mock_validator)
        validator = container.data_validator()
        assert validator is mock_validator

    def test_feature_engineer_lazy_creation(self):
        """Test that feature_engineer is created lazily."""
        container = Container.from_settings()
        engineer = container.feature_engineer()
        assert isinstance(engineer, FeatureEngineer)

    def test_preprocessor_lazy_creation(self):
        """Test that preprocessor is created lazily."""
        container = Container.from_settings()
        preprocessor = container.preprocessor()
        assert isinstance(preprocessor, Preprocessor)

    def test_for_testing_creates_testing_container(self):
        """Test that for_testing creates container with testing env."""
        container = Container.for_testing()
        assert container.settings.environment == "testing"

    def test_for_testing_with_mocks(self):
        """Test that for_testing accepts mock dependencies."""
        mock_loader = MagicMock(spec=DataLoader)
        mock_validator = MagicMock(spec=DataValidator)
        mock_engineer = MagicMock(spec=FeatureEngineer)
        mock_preprocessor = MagicMock(spec=Preprocessor)

        container = Container.for_testing(
            data_loader=mock_loader,
            validator=mock_validator,
            feature_engineer=mock_engineer,
            preprocessor=mock_preprocessor,
        )

        assert container.data_loader() is mock_loader
        assert container.data_validator() is mock_validator
        assert container.feature_engineer() is mock_engineer
        assert container.preprocessor() is mock_preprocessor


class TestGetContainer:
    """Tests for get_container function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        clear_settings_cache()
        reset_container()
        yield
        clear_settings_cache()
        reset_container()

    def test_returns_container(self):
        """Test that get_container returns a Container."""
        container = get_container()
        assert isinstance(container, Container)

    def test_returns_same_container(self):
        """Test that get_container returns cached container."""
        container1 = get_container()
        container2 = get_container()
        assert container1 is container2

    def test_reset_creates_new_container(self):
        """Test that reset=True creates a new container."""
        container1 = get_container()
        container2 = get_container(reset=True)
        assert container1 is not container2


class TestContainerIntegration:
    """Integration tests for Container with pipelines."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        clear_settings_cache()
        reset_container()
        yield
        clear_settings_cache()
        reset_container()

    def test_training_pipeline_creation(self):
        """Test that training_pipeline creates pipeline with dependencies."""
        container = Container.from_settings()

        # Mock the actual model training to avoid MLflow setup
        with patch.object(container, "model_trainer") as mock_trainer_method:
            mock_trainer = MagicMock(spec=ModelTrainer)
            mock_trainer_method.return_value = mock_trainer

            pipeline = container.training_pipeline()

            assert pipeline is not None
            assert pipeline.data_loader is not None
            assert pipeline.data_validator is not None
            assert pipeline.feature_engineer is not None
            assert pipeline.preprocessor is not None
