"""
Settings Module
===============

Pydantic-based configuration management with support for:
- Environment variables
- .env files
- YAML configuration files
- Environment-specific overrides (dev, staging, prod)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Data-related configuration."""

    raw_path: str = "data/raw/Hotel Reservations.csv"
    processed_path: str = "data/processed"

    # Validation settings
    expected_columns: List[str] = [
        "Booking_ID",
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "type_of_meal_plan",
        "required_car_parking_space",
        "room_type_reserved",
        "lead_time",
        "arrival_year",
        "arrival_month",
        "arrival_date",
        "market_segment_type",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
        "booking_status",
    ]


class ModelConfig(BaseModel):
    """Model-related configuration."""

    name: str = "hotel_cancellation_model"
    algorithm: str = "xgboost"

    # XGBoost default parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    eval_metric: str = "logloss"

    # Feature engineering
    target_column: str = "booking_status"
    columns_to_drop: List[str] = ["Booking_ID"]
    target_mapping: Dict[str, int] = {"Canceled": 1, "Not_Canceled": 0}

    @property
    def xgboost_params(self) -> Dict[str, Any]:
        """Return XGBoost parameters as dictionary."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "eval_metric": self.eval_metric,
        }


class TrainingConfig(BaseModel):
    """Training-related configuration."""

    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    random_state: int = 42
    stratify: bool = True


class DatabaseConfig(BaseModel):
    """Database configuration for MLflow backend store."""

    # Supabase PostgreSQL connection string
    # Format: postgresql://postgres.[PROJECT_REF]:[PASSWORD]@aws-0-[REGION]...
    supabase_url: Optional[str] = Field(
        default=None,
        description="Supabase PostgreSQL connection string",
    )

    # Fallback to SQLite for local development
    sqlite_path: str = "sqlite:///mlflow/db/mlflow.db"

    @property
    def backend_store_uri(self) -> str:
        """Return the appropriate backend store URI."""
        if self.supabase_url:
            return self.supabase_url
        return self.sqlite_path

    @property
    def is_cloud_db(self) -> bool:
        """Check if using cloud database (Supabase)."""
        return self.supabase_url is not None


class MLflowConfig(BaseModel):
    """MLflow-related configuration."""

    experiment_name: str = "hotel_cancellation_prediction"
    tracking_uri: Optional[str] = None
    registry_uri: Optional[str] = None
    model_name: str = "hotel_cancellation_model"
    artifact_root: str = "/mlflow/artifacts"


class ArtifactsConfig(BaseModel):
    """Artifacts storage configuration."""

    models_dir: str = "artifacts/models"
    preprocessor_filename: str = "preprocessor.joblib"
    feature_engineer_filename: str = "feature_engineer.joblib"

    @property
    def preprocessor_path(self) -> str:
        return f"{self.models_dir}/{self.preprocessor_filename}"

    @property
    def feature_engineer_path(self) -> str:
        return f"{self.models_dir}/{self.feature_engineer_filename}"


class APIConfig(BaseModel):
    """API-related configuration."""

    host: str = "0.0.0.0"  # nosec: B104
    port: int = 8000
    workers: int = 1
    reload: bool = False

    # CORS settings
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Model serving
    model_uri: Optional[str] = None


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    json_format: bool = False


class FeatureStoreConfig(BaseModel):
    """Feature Store configuration."""

    enabled: bool = Field(default=True, description="Enable/disable Feature Store")
    db_path: str = Field(
        default="feature_store.db",
        description="Path to Feature Store database (SQLite)",
    )
    auto_store: bool = Field(
        default=True,
        description="Automatically store features during training",
    )
    auto_retrieve: bool = Field(
        default=True,
        description="Automatically retrieve stored features during inference",
    )


class Settings(BaseSettings):
    """
    Main settings class that aggregates all configuration.

    Configuration priority (highest to lowest):
    1. Environment variables
    2. .env file
    3. YAML config file
    4. Default values
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Environment
    environment: str = Field(default="development", description="Runtime environment")
    debug: bool = Field(default=False, description="Debug mode")

    # Sub-configurations
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        allowed = ["development", "staging", "production", "testing"]
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v.lower()

    @classmethod
    def from_yaml(cls, yaml_path: str, env_override: bool = True) -> "Settings":
        """
        Load settings from YAML file with optional environment override.

        Priority (highest to lowest):
        1. Environment variables (when env_override=True)
        2. YAML file values
        3. Default values

        Args:
            yaml_path: Path to YAML configuration file.
            env_override: If True, environment variables override YAML values.

        Returns:
            Settings instance.
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

        if env_override:
            # First create settings from env vars and defaults
            env_settings = cls()

            # Then merge: env vars > yaml > defaults
            # We need to deep merge the configs
            merged_config = cls._deep_merge_with_env(yaml_config, env_settings)
            return cls.model_validate(merged_config)
        else:
            # YAML values take precedence (disable env loading)
            return cls.model_validate(yaml_config)

    @classmethod
    def _deep_merge_with_env(cls, yaml_config: dict, env_settings: "Settings") -> dict:
        """
        Deep merge YAML config with environment settings.

        Environment variables take precedence over YAML values.
        """
        result = {}

        # Get env settings as dict
        env_dict = env_settings.model_dump()

        for key, yaml_value in yaml_config.items():
            env_value = env_dict.get(key)

            if isinstance(yaml_value, dict) and isinstance(env_value, dict):
                # For nested configs, check each field
                merged_nested = {}
                for nested_key, nested_yaml_val in yaml_value.items():
                    nested_env_val = env_value.get(nested_key)
                    # Check if env var was explicitly set (different from default)
                    env_var_name = f"{key.upper()}__{nested_key.upper()}"
                    if os.getenv(env_var_name) is not None:
                        # Env var was set, use env value
                        merged_nested[nested_key] = nested_env_val
                    else:
                        # Use YAML value
                        merged_nested[nested_key] = nested_yaml_val
                result[key] = merged_nested
            else:
                # For top-level values, check env var
                env_var_name = key.upper()
                if os.getenv(env_var_name) is not None:
                    result[key] = env_value
                else:
                    result[key] = yaml_value

        # Add any keys from env that aren't in YAML
        for key, env_value in env_dict.items():
            if key not in result:
                result[key] = env_value

        return result

    def get_model_uri(self) -> str:
        """Get the model URI, with fallback to default."""
        if self.api.model_uri:
            return self.api.model_uri
        return f"models:/{self.mlflow.model_name}/latest"

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Singleton pattern with caching
@lru_cache()
def get_settings(
    config_path: Optional[str] = None,
    env: Optional[str] = None,
) -> Settings:
    """
    Get application settings (cached singleton).

    Args:
        config_path: Optional path to YAML config file.
        env: Optional environment override.

    Returns:
        Settings instance.

    Usage:
        # Default settings from env vars
        settings = get_settings()

        # From YAML file
        settings = get_settings(config_path="configs/config.yaml")

        # With environment override
        settings = get_settings(env="production")
    """
    # Check for config path in environment
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH")

    # Check for environment override
    if env is not None:
        os.environ["ENVIRONMENT"] = env

    if config_path and Path(config_path).exists():
        return Settings.from_yaml(config_path)

    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache (useful for testing)."""
    get_settings.cache_clear()
