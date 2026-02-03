"""
Unit Tests for API
==================

Tests for the FastAPI application endpoints.
"""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.api.app import AppState, create_app


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    @pytest.fixture
    def mock_inference_pipeline(self):
        """Create a mock inference pipeline."""
        pipeline = Mock()
        pipeline.is_loaded = False
        pipeline.model_uri = "models:/test/1"
        return pipeline

    @pytest.fixture
    def loaded_mock_pipeline(self, mock_inference_pipeline):
        """Create a loaded mock inference pipeline."""
        mock_inference_pipeline.is_loaded = True
        mock_inference_pipeline.predict_single.return_value = {
            "prediction": 0,
            "probability": 0.25,
            "label": "Not Canceled",
        }
        mock_inference_pipeline.predict_batch.return_value = [
            {"prediction": 0, "probability": 0.25, "label": "Not Canceled"},
            {"prediction": 1, "probability": 0.75, "label": "Canceled"},
        ]
        return mock_inference_pipeline

    @pytest.fixture
    def client_without_model(self, mock_inference_pipeline) -> TestClient:
        """Create a test client with unloaded model."""
        app = create_app()

        # Initialize app state manually for testing
        app_state = AppState()
        app_state.inference_pipeline = mock_inference_pipeline
        app.state.app_state = app_state

        return TestClient(app, raise_server_exceptions=False)

    @pytest.fixture
    def client_with_model(self, loaded_mock_pipeline) -> TestClient:
        """Create a test client with loaded model."""
        app = create_app()

        # Initialize app state manually for testing
        app_state = AppState()
        app_state.inference_pipeline = loaded_mock_pipeline
        app.state.app_state = app_state

        return TestClient(app, raise_server_exceptions=False)

    @pytest.fixture
    def sample_reservation(self) -> dict:
        """Sample reservation data for testing."""
        return {
            "no_of_adults": 2,
            "no_of_children": 0,
            "no_of_weekend_nights": 1,
            "no_of_week_nights": 2,
            "type_of_meal_plan": "Meal Plan 1",
            "required_car_parking_space": 0,
            "room_type_reserved": "Room_Type 1",
            "lead_time": 224,
            "arrival_year": 2018,
            "arrival_month": 10,
            "arrival_date": 2,
            "market_segment_type": "Online",
            "repeated_guest": 0,
            "no_of_previous_cancellations": 0,
            "no_of_previous_bookings_not_canceled": 0,
            "avg_price_per_room": 65.0,
            "no_of_special_requests": 0,
        }

    def test_root_endpoint(self, client_without_model: TestClient):
        """Test the root endpoint returns API info."""
        response = client_without_model.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data

    def test_health_endpoint_degraded(self, client_without_model: TestClient):
        """Test the health check endpoint when model not loaded."""
        response = client_without_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

    def test_health_endpoint_healthy(self, client_with_model: TestClient):
        """Test the health check endpoint when model is loaded."""
        response = client_with_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_predict_without_model_returns_503(
        self, client_without_model: TestClient, sample_reservation: dict
    ):
        """Test prediction endpoint returns 503 when model not loaded."""
        response = client_without_model.post("/predict", json=sample_reservation)
        assert response.status_code == 503

    def test_predict_batch_without_model_returns_503(
        self, client_without_model: TestClient, sample_reservation: dict
    ):
        """Test batch prediction returns 503 when model not loaded."""
        batch_request = {"reservations": [sample_reservation, sample_reservation]}
        response = client_without_model.post("/predict/batch", json=batch_request)
        assert response.status_code == 503

    def test_predict_success(
        self, client_with_model: TestClient, sample_reservation: dict
    ):
        """Test successful single prediction."""
        response = client_with_model.post("/predict", json=sample_reservation)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "label" in data
        assert data["prediction"] == 0
        assert data["label"] == "Not Canceled"

    def test_predict_batch_success(
        self, client_with_model: TestClient, sample_reservation: dict
    ):
        """Test successful batch prediction."""
        batch_request = {"reservations": [sample_reservation, sample_reservation]}
        response = client_with_model.post("/predict/batch", json=batch_request)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_predict_invalid_input(self, client_with_model: TestClient):
        """Test prediction with invalid input returns 422."""
        invalid_data = {
            "no_of_adults": "not_a_number",  # Invalid type
            "no_of_children": 0,
        }
        response = client_with_model.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_missing_required_fields(self, client_with_model: TestClient):
        """Test prediction with missing required fields returns 422."""
        incomplete_data = {
            "no_of_adults": 2,
            # Missing many required fields
        }
        response = client_with_model.post("/predict", json=incomplete_data)
        assert response.status_code == 422

    def test_batch_predict_empty_list(self, client_with_model: TestClient):
        """Test batch prediction with empty list returns 422."""
        empty_batch = {"reservations": []}
        response = client_with_model.post("/predict/batch", json=empty_batch)
        assert response.status_code == 422

    def test_predict_negative_adults(self, client_with_model: TestClient):
        """Test prediction rejects negative adults value."""
        data = {
            "no_of_adults": -1,  # Invalid: negative
            "no_of_children": 0,
            "no_of_weekend_nights": 1,
            "no_of_week_nights": 2,
            "type_of_meal_plan": "Meal Plan 1",
            "required_car_parking_space": 0,
            "room_type_reserved": "Room_Type 1",
            "lead_time": 224,
            "arrival_year": 2018,
            "arrival_month": 10,
            "arrival_date": 2,
            "market_segment_type": "Online",
            "repeated_guest": 0,
            "no_of_previous_cancellations": 0,
            "no_of_previous_bookings_not_canceled": 0,
            "avg_price_per_room": 65.0,
            "no_of_special_requests": 0,
        }
        response = client_with_model.post("/predict", json=data)
        assert response.status_code == 422

    def test_predict_invalid_month(self, client_with_model: TestClient):
        """Test prediction rejects invalid month value."""
        data = {
            "no_of_adults": 2,
            "no_of_children": 0,
            "no_of_weekend_nights": 1,
            "no_of_week_nights": 2,
            "type_of_meal_plan": "Meal Plan 1",
            "required_car_parking_space": 0,
            "room_type_reserved": "Room_Type 1",
            "lead_time": 224,
            "arrival_year": 2018,
            "arrival_month": 15,  # Invalid: > 12
            "arrival_date": 2,
            "market_segment_type": "Online",
            "repeated_guest": 0,
            "no_of_previous_cancellations": 0,
            "no_of_previous_bookings_not_canceled": 0,
            "avg_price_per_room": 65.0,
            "no_of_special_requests": 0,
        }
        response = client_with_model.post("/predict", json=data)
        assert response.status_code == 422
