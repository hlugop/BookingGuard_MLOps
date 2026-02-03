"""
Unit Tests for Custom Exceptions
================================

Tests for the custom exception classes.
"""

from src.exceptions import (
    APIError,
    ArtifactNotFoundError,
    BadRequestError,
    BaseMLOpsException,
    DataLoadError,
    DataValidationError,
    MissingColumnError,
    ModelError,
    ModelLoadError,
    ModelNotLoadedError,
    PipelineError,
    PipelineNotLoadedError,
    PredictionError,
    ServiceUnavailableError,
)


class TestBaseMLOpsException:
    """Tests for BaseMLOpsException."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        exc = BaseMLOpsException("Test error")
        assert exc.message == "Test error"
        assert exc.details == {}
        assert exc.original_error is None

    def test_exception_with_details(self):
        """Test exception with details."""
        exc = BaseMLOpsException(
            "Test error",
            details={"key": "value"},
        )
        assert exc.details == {"key": "value"}

    def test_exception_with_original_error(self):
        """Test exception wrapping another exception."""
        original = ValueError("Original error")
        exc = BaseMLOpsException(
            "Wrapped error",
            original_error=original,
        )
        assert exc.original_error is original

    def test_to_dict(self):
        """Test to_dict method."""
        exc = BaseMLOpsException(
            "Test error",
            details={"key": "value"},
        )
        result = exc.to_dict()
        assert result["error"] == "BaseMLOpsException"
        assert result["message"] == "Test error"
        assert result["details"] == {"key": "value"}

    def test_to_dict_without_details(self):
        """Test to_dict method without details."""
        exc = BaseMLOpsException("Test error")
        result = exc.to_dict()
        assert "details" not in result


class TestDataExceptions:
    """Tests for data-related exceptions."""

    def test_data_load_error(self):
        """Test DataLoadError."""
        exc = DataLoadError(path="/path/to/file.csv")
        assert "Failed to load data" in exc.message
        assert exc.details["path"] == "/path/to/file.csv"

    def test_data_validation_error(self):
        """Test DataValidationError."""
        errors = ["Error 1", "Error 2"]
        exc = DataValidationError(errors=errors)
        assert "validation failed" in exc.message.lower()
        assert exc.details["validation_errors"] == errors

    def test_missing_column_error(self):
        """Test MissingColumnError."""
        missing = ["col1", "col2"]
        exc = MissingColumnError(missing_columns=missing)
        assert "col1" in exc.message
        assert "col2" in exc.message
        assert exc.missing_columns == missing


class TestModelExceptions:
    """Tests for model-related exceptions."""

    def test_model_not_loaded_error(self):
        """Test ModelNotLoadedError."""
        exc = ModelNotLoadedError(model_uri="models:/test/v1")
        assert "not loaded" in exc.message.lower()
        assert exc.details["model_uri"] == "models:/test/v1"

    def test_model_load_error(self):
        """Test ModelLoadError."""
        exc = ModelLoadError(
            message="Connection failed",
            model_uri="models:/test/v1",
        )
        assert exc.message == "Connection failed"
        assert exc.details["model_uri"] == "models:/test/v1"

    def test_prediction_error(self):
        """Test PredictionError."""
        exc = PredictionError(
            message="Invalid input",
            input_shape=(100, 20),
        )
        assert exc.message == "Invalid input"
        assert exc.details["input_shape"] == (100, 20)


class TestPipelineExceptions:
    """Tests for pipeline-related exceptions."""

    def test_pipeline_not_loaded_error(self):
        """Test PipelineNotLoadedError."""
        exc = PipelineNotLoadedError()
        assert "not loaded" in exc.message.lower()
        assert "load()" in exc.message

    def test_artifact_not_found_error(self):
        """Test ArtifactNotFoundError."""
        exc = ArtifactNotFoundError(
            artifact_name="preprocessor",
            artifact_path="/path/to/preprocessor.joblib",
        )
        assert "preprocessor" in exc.message.lower()
        assert exc.details["artifact_name"] == "preprocessor"
        assert exc.details["artifact_path"] == "/path/to/preprocessor.joblib"


class TestAPIExceptions:
    """Tests for API-related exceptions."""

    def test_api_error_default_status(self):
        """Test APIError with default status code."""
        exc = APIError("Something went wrong")
        assert exc.status_code == 500

    def test_api_error_custom_status(self):
        """Test APIError with custom status code."""
        exc = APIError("Not found", status_code=404)
        assert exc.status_code == 404

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError."""
        exc = ServiceUnavailableError()
        assert exc.status_code == 503

    def test_bad_request_error(self):
        """Test BadRequestError."""
        exc = BadRequestError("Invalid input")
        assert exc.status_code == 400
        assert exc.message == "Invalid input"


class TestExceptionInheritance:
    """Tests for exception inheritance hierarchy."""

    def test_data_exceptions_inherit_from_base(self):
        """Test that data exceptions inherit from BaseMLOpsException."""
        assert issubclass(DataLoadError, BaseMLOpsException)
        assert issubclass(DataValidationError, BaseMLOpsException)
        assert issubclass(MissingColumnError, DataValidationError)

    def test_model_exceptions_inherit_from_base(self):
        """Test that model exceptions inherit from BaseMLOpsException."""
        assert issubclass(ModelError, BaseMLOpsException)
        assert issubclass(ModelNotLoadedError, ModelError)
        assert issubclass(ModelLoadError, ModelError)
        assert issubclass(PredictionError, ModelError)

    def test_pipeline_exceptions_inherit_from_base(self):
        """Test that pipeline exceptions inherit from BaseMLOpsException."""
        assert issubclass(PipelineError, BaseMLOpsException)
        assert issubclass(PipelineNotLoadedError, PipelineError)
        assert issubclass(ArtifactNotFoundError, PipelineError)

    def test_api_exceptions_inherit_from_base(self):
        """Test that API exceptions inherit from BaseMLOpsException."""
        assert issubclass(APIError, BaseMLOpsException)
        assert issubclass(ServiceUnavailableError, APIError)
        assert issubclass(BadRequestError, APIError)
