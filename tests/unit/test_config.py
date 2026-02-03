"""
Unit Tests for Configuration
============================

Tests for the Settings and configuration management.
"""

import pytest

from src.config.settings import (
    DatabaseConfig,
    DataConfig,
    ModelConfig,
    Settings,
    TrainingConfig,
    clear_settings_cache,
    get_settings,
)


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.raw_path == "data/raw/Hotel Reservations.csv"
        assert config.processed_path == "data/processed"
        assert len(config.expected_columns) == 19

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DataConfig(
            raw_path="custom/path.csv",
            processed_path="custom/processed",
        )
        assert config.raw_path == "custom/path.csv"
        assert config.processed_path == "custom/processed"


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.name == "hotel_cancellation_model"
        assert config.algorithm == "xgboost"
        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.1

    def test_xgboost_params_property(self):
        """Test xgboost_params property returns correct dict."""
        config = ModelConfig(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
        )
        params = config.xgboost_params
        assert params["n_estimators"] == 200
        assert params["max_depth"] == 10
        assert params["learning_rate"] == 0.05
        assert params["eval_metric"] == "logloss"


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.stratify is True

    def test_test_size_validation(self):
        """Test that test_size must be between 0.1 and 0.5."""
        # Valid values
        config = TrainingConfig(test_size=0.3)
        assert config.test_size == 0.3

        # Invalid: too low
        with pytest.raises(ValueError):
            TrainingConfig(test_size=0.05)

        # Invalid: too high
        with pytest.raises(ValueError):
            TrainingConfig(test_size=0.6)


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_values(self):
        """Test default database configuration uses SQLite."""
        config = DatabaseConfig()
        assert config.supabase_url is None
        assert config.sqlite_path == "sqlite:///mlflow/db/mlflow.db"

    def test_backend_store_uri_sqlite_default(self):
        """Test backend_store_uri returns SQLite when no Supabase URL set."""
        config = DatabaseConfig()
        assert config.backend_store_uri == "sqlite:///mlflow/db/mlflow.db"
        assert config.is_cloud_db is False

    def test_backend_store_uri_supabase(self):
        """Test backend_store_uri returns Supabase URL when set."""
        supabase_url = "postgresql://postgres.abc123:password@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
        config = DatabaseConfig(supabase_url=supabase_url)
        assert config.backend_store_uri == supabase_url
        assert config.is_cloud_db is True

    def test_is_cloud_db_false_by_default(self):
        """Test is_cloud_db returns False when using SQLite."""
        config = DatabaseConfig()
        assert config.is_cloud_db is False

    def test_is_cloud_db_true_with_supabase(self):
        """Test is_cloud_db returns True when Supabase URL is set."""
        config = DatabaseConfig(
            supabase_url="postgresql://user:pass@host:5432/db"
        )
        assert config.is_cloud_db is True

    def test_custom_sqlite_path(self):
        """Test custom SQLite path is used when no Supabase URL."""
        config = DatabaseConfig(sqlite_path="sqlite:///custom/path/mlflow.db")
        assert config.backend_store_uri == "sqlite:///custom/path/mlflow.db"
        assert config.is_cloud_db is False

    def test_supabase_takes_precedence_over_sqlite(self):
        """Test Supabase URL takes precedence over SQLite path."""
        config = DatabaseConfig(
            supabase_url="postgresql://user:pass@host:5432/db",
            sqlite_path="sqlite:///custom/path.db",
        )
        assert config.backend_store_uri == "postgresql://user:pass@host:5432/db"
        assert config.is_cloud_db is True


class TestSettings:
    """Tests for main Settings class."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear settings cache before each test."""
        clear_settings_cache()
        yield
        clear_settings_cache()

    def test_default_settings(self):
        """Test default settings are created correctly."""
        settings = Settings()
        assert settings.environment == "development"
        assert settings.debug is False
        assert isinstance(settings.data, DataConfig)
        assert isinstance(settings.model, ModelConfig)
        assert isinstance(settings.training, TrainingConfig)
        assert isinstance(settings.database, DatabaseConfig)

    def test_settings_database_default(self):
        """Test Settings includes DatabaseConfig with SQLite default."""
        settings = Settings()
        assert settings.database.supabase_url is None
        assert settings.database.is_cloud_db is False
        assert "sqlite" in settings.database.backend_store_uri

    def test_settings_database_with_supabase(self):
        """Test Settings with Supabase configuration."""
        db_config = DatabaseConfig(
            supabase_url="postgresql://user:pass@host:5432/db"
        )
        settings = Settings(database=db_config)
        assert settings.database.is_cloud_db is True
        assert settings.database.backend_store_uri == "postgresql://user:pass@host:5432/db"

    def test_environment_validation(self):
        """Test environment must be valid."""
        # Valid environments
        for env in ["development", "staging", "production", "testing"]:
            settings = Settings(environment=env)
            assert settings.environment == env

        # Invalid environment
        with pytest.raises(ValueError):
            Settings(environment="invalid")

    def test_is_production(self):
        """Test is_production method."""
        dev_settings = Settings(environment="development")
        assert dev_settings.is_production() is False

        prod_settings = Settings(environment="production")
        assert prod_settings.is_production() is True

    def test_is_development(self):
        """Test is_development method."""
        dev_settings = Settings(environment="development")
        assert dev_settings.is_development() is True

        prod_settings = Settings(environment="production")
        assert prod_settings.is_development() is False

    def test_get_model_uri_default(self):
        """Test get_model_uri returns default when not set."""
        settings = Settings()
        uri = settings.get_model_uri()
        assert uri == "models:/hotel_cancellation_model/latest"

    def test_get_model_uri_custom(self):
        """Test get_model_uri returns custom URI when set."""
        from src.config.settings import APIConfig

        settings = Settings(api=APIConfig(model_uri="models:/custom_model/v1"))
        uri = settings.get_model_uri()
        assert uri == "models:/custom_model/v1"


class TestGetSettings:
    """Tests for get_settings function."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear settings cache before each test."""
        clear_settings_cache()
        yield
        clear_settings_cache()

    def test_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_cached_settings(self):
        """Test that settings are cached."""
        settings1 = get_settings()
        settings2 = get_settings()
        # Should return the same cached instance
        assert settings1 is settings2

    def test_env_override(self):
        """Test environment override."""
        settings = get_settings(env="production")
        assert settings.environment == "production"
