"""
Custom Exceptions Module
========================

Defines application-specific exceptions for better error handling
and clearer error messages.
"""

from typing import Any, Dict, List, Optional


class BaseMLOpsException(Exception):
    """Base exception for all MLOps-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result = {
            "error": self.__class__.__name__,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# ============================================================================
# Data Exceptions
# ============================================================================


class DataLoadError(BaseMLOpsException):
    """Raised when data cannot be loaded."""

    def __init__(
        self,
        message: str = "Failed to load data",
        path: Optional[str] = None,
        **kwargs,
    ):
        details = {"path": path} if path else {}
        super().__init__(message, details=details, **kwargs)


class DataValidationError(BaseMLOpsException):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str = "Data validation failed",
        errors: Optional[List[str]] = None,
        **kwargs,
    ):
        details = {"validation_errors": errors} if errors else {}
        super().__init__(message, details=details, **kwargs)


class MissingColumnError(DataValidationError):
    """Raised when required columns are missing."""

    def __init__(
        self,
        missing_columns: List[str],
        message: Optional[str] = None,
        **kwargs,
    ):
        msg = message or f"Missing required columns: {missing_columns}"
        super().__init__(
            msg, errors=[f"Missing: {col}" for col in missing_columns], **kwargs
        )
        self.missing_columns = missing_columns


# ============================================================================
# Feature Engineering Exceptions
# ============================================================================


class FeatureEngineeringError(BaseMLOpsException):
    """Raised when feature engineering fails."""


class PreprocessorNotFittedError(FeatureEngineeringError):
    """Raised when using a preprocessor that hasn't been fitted."""

    def __init__(
        self,
        message: str = "Preprocessor must be fitted before transform",
        **kwargs,
    ):
        super().__init__(message, **kwargs)


class FeatureEngineerNotFittedError(FeatureEngineeringError):
    """Raised when using a feature engineer that hasn't been fitted."""

    def __init__(
        self,
        message: str = "FeatureEngineer must be fitted before accessing feature names",
        **kwargs,
    ):
        super().__init__(message, **kwargs)


# ============================================================================
# Model Exceptions
# ============================================================================


class ModelError(BaseMLOpsException):
    """Base exception for model-related errors."""


class ModelNotLoadedError(ModelError):
    """Raised when attempting to use a model that hasn't been loaded."""

    def __init__(
        self,
        message: str = "Model not loaded. Call load_model() first.",
        model_uri: Optional[str] = None,
        **kwargs,
    ):
        details = {"model_uri": model_uri} if model_uri else {}
        super().__init__(message, details=details, **kwargs)


class ModelLoadError(ModelError):
    """Raised when model loading fails."""

    def __init__(
        self,
        message: str = "Failed to load model",
        model_uri: Optional[str] = None,
        **kwargs,
    ):
        details = {"model_uri": model_uri} if model_uri else {}
        super().__init__(message, details=details, **kwargs)


class ModelTrainingError(ModelError):
    """Raised when model training fails."""


class PredictionError(ModelError):
    """Raised when prediction fails."""

    def __init__(
        self,
        message: str = "Prediction failed",
        input_shape: Optional[tuple] = None,
        **kwargs,
    ):
        details = {"input_shape": input_shape} if input_shape else {}
        super().__init__(message, details=details, **kwargs)


# ============================================================================
# Pipeline Exceptions
# ============================================================================


class PipelineError(BaseMLOpsException):
    """Base exception for pipeline-related errors."""


class PipelineNotLoadedError(PipelineError):
    """Raised when pipeline is not properly initialized."""

    def __init__(
        self,
        message: str = "Pipeline not loaded. Call load() first.",
        **kwargs,
    ):
        super().__init__(message, **kwargs)


class ArtifactNotFoundError(PipelineError):
    """Raised when a required artifact is not found."""

    def __init__(
        self,
        artifact_name: str,
        artifact_path: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs,
    ):
        msg = message or f"Artifact not found: {artifact_name}"
        details = {
            "artifact_name": artifact_name,
            "artifact_path": artifact_path,
        }
        super().__init__(msg, details=details, **kwargs)


# ============================================================================
# Configuration Exceptions
# ============================================================================


class ConfigurationError(BaseMLOpsException):
    """Raised when there's a configuration error."""


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""

    def __init__(
        self,
        message: str = "Invalid configuration",
        config_key: Optional[str] = None,
        **kwargs,
    ):
        details = {"config_key": config_key} if config_key else {}
        super().__init__(message, details=details, **kwargs)


# ============================================================================
# API Exceptions
# ============================================================================


class APIError(BaseMLOpsException):
    """Base exception for API-related errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code


class ServiceUnavailableError(APIError):
    """Raised when service is unavailable (503)."""

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        **kwargs,
    ):
        super().__init__(message, status_code=503, **kwargs)


class BadRequestError(APIError):
    """Raised for bad request errors (400)."""

    def __init__(
        self,
        message: str = "Bad request",
        **kwargs,
    ):
        super().__init__(message, status_code=400, **kwargs)


# ============================================================================
# Feature Store Exceptions
# ============================================================================


class FeatureStoreError(BaseMLOpsException):
    """Base exception for Feature Store errors."""


class FeatureVersionNotFoundError(FeatureStoreError):
    """Raised when a feature version is not found."""

    def __init__(
        self,
        version: str,
        message: Optional[str] = None,
        **kwargs,
    ):
        msg = message or f"Feature version not found: {version}"
        details = {"version": version}
        super().__init__(msg, details=details, **kwargs)


class FeatureStoreConnectionError(FeatureStoreError):
    """Raised when Feature Store cannot connect to database."""

    def __init__(
        self,
        message: str = "Failed to connect to Feature Store database",
        **kwargs,
    ):
        super().__init__(message, **kwargs)
