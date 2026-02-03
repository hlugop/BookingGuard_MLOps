"""
Unit Tests for Data Validator
=============================

Tests for the DataValidator class.
"""

import pandas as pd
import pytest

from src.data.validator import DataValidator, ValidationResult


class TestDataValidator:
    """Test suite for DataValidator."""

    @pytest.fixture
    def validator(self) -> DataValidator:
        """Create a DataValidator instance."""
        return DataValidator()

    @pytest.fixture
    def valid_data(self) -> pd.DataFrame:
        """Create valid sample data."""
        return pd.DataFrame(
            {
                "Booking_ID": ["INN00001", "INN00002"],
                "no_of_adults": [2, 1],
                "no_of_children": [0, 1],
                "no_of_weekend_nights": [1, 2],
                "no_of_week_nights": [2, 3],
                "type_of_meal_plan": ["Meal Plan 1", "Meal Plan 2"],
                "required_car_parking_space": [0, 1],
                "room_type_reserved": ["Room_Type 1", "Room_Type 2"],
                "lead_time": [224, 5],
                "arrival_year": [2018, 2018],
                "arrival_month": [10, 11],
                "arrival_date": [2, 6],
                "market_segment_type": ["Online", "Offline"],
                "repeated_guest": [0, 1],
                "no_of_previous_cancellations": [0, 0],
                "no_of_previous_bookings_not_canceled": [0, 2],
                "avg_price_per_room": [65.0, 106.68],
                "no_of_special_requests": [0, 1],
                "booking_status": ["Not_Canceled", "Canceled"],
            }
        )

    def test_validate_valid_data(
        self, validator: DataValidator, valid_data: pd.DataFrame
    ):
        """Test validation passes for valid data."""
        result = validator.validate(valid_data)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_returns_stats(
        self, validator: DataValidator, valid_data: pd.DataFrame
    ):
        """Test that validation returns statistics."""
        result = validator.validate(valid_data)
        assert "total_records" in result.stats
        assert result.stats["total_records"] == 2
        assert "target_distribution" in result.stats

    def test_validate_invalid_booking_status(self, validator: DataValidator):
        """Test validation fails for invalid booking status."""
        invalid_data = pd.DataFrame(
            {
                "Booking_ID": ["INN00001"],
                "no_of_adults": [2],
                "no_of_children": [0],
                "no_of_weekend_nights": [1],
                "no_of_week_nights": [2],
                "type_of_meal_plan": ["Meal Plan 1"],
                "required_car_parking_space": [0],
                "room_type_reserved": ["Room_Type 1"],
                "lead_time": [224],
                "arrival_year": [2018],
                "arrival_month": [10],
                "arrival_date": [2],
                "market_segment_type": ["Online"],
                "repeated_guest": [0],
                "no_of_previous_cancellations": [0],
                "no_of_previous_bookings_not_canceled": [0],
                "avg_price_per_room": [65.0],
                "no_of_special_requests": [0],
                "booking_status": ["Invalid_Status"],  # Invalid
            }
        )
        result = validator.validate(invalid_data)
        assert result.is_valid is False
        assert any("booking_status" in error for error in result.errors)

    def test_validate_negative_values(self, validator: DataValidator):
        """Test validation detects negative values in non-negative columns."""
        data = pd.DataFrame(
            {
                "Booking_ID": ["INN00001"],
                "no_of_adults": [-1],  # Invalid: negative
                "no_of_children": [0],
                "no_of_weekend_nights": [1],
                "no_of_week_nights": [2],
                "type_of_meal_plan": ["Meal Plan 1"],
                "required_car_parking_space": [0],
                "room_type_reserved": ["Room_Type 1"],
                "lead_time": [224],
                "arrival_year": [2018],
                "arrival_month": [10],
                "arrival_date": [2],
                "market_segment_type": ["Online"],
                "repeated_guest": [0],
                "no_of_previous_cancellations": [0],
                "no_of_previous_bookings_not_canceled": [0],
                "avg_price_per_room": [65.0],
                "no_of_special_requests": [0],
                "booking_status": ["Not_Canceled"],
            }
        )
        result = validator.validate(data)
        assert result.is_valid is False
        assert any("negative" in error.lower() for error in result.errors)

    def test_validate_invalid_month(self, validator: DataValidator):
        """Test validation detects invalid month values."""
        data = pd.DataFrame(
            {
                "Booking_ID": ["INN00001"],
                "no_of_adults": [2],
                "no_of_children": [0],
                "no_of_weekend_nights": [1],
                "no_of_week_nights": [2],
                "type_of_meal_plan": ["Meal Plan 1"],
                "required_car_parking_space": [0],
                "room_type_reserved": ["Room_Type 1"],
                "lead_time": [224],
                "arrival_year": [2018],
                "arrival_month": [13],  # Invalid: > 12
                "arrival_date": [2],
                "market_segment_type": ["Online"],
                "repeated_guest": [0],
                "no_of_previous_cancellations": [0],
                "no_of_previous_bookings_not_canceled": [0],
                "avg_price_per_room": [65.0],
                "no_of_special_requests": [0],
                "booking_status": ["Not_Canceled"],
            }
        )
        result = validator.validate(data)
        assert result.is_valid is False
        assert any("arrival_month" in error for error in result.errors)

    def test_validate_missing_values_warning(self, validator: DataValidator):
        """Test validation warns about missing values."""
        data = pd.DataFrame(
            {
                "Booking_ID": ["INN00001", "INN00002"],
                "no_of_adults": [2, None],  # Missing value
                "no_of_children": [0, 0],
                "no_of_weekend_nights": [1, 2],
                "no_of_week_nights": [2, 3],
                "type_of_meal_plan": ["Meal Plan 1", "Meal Plan 2"],
                "required_car_parking_space": [0, 1],
                "room_type_reserved": ["Room_Type 1", "Room_Type 2"],
                "lead_time": [224, 5],
                "arrival_year": [2018, 2018],
                "arrival_month": [10, 11],
                "arrival_date": [2, 6],
                "market_segment_type": ["Online", "Offline"],
                "repeated_guest": [0, 1],
                "no_of_previous_cancellations": [0, 0],
                "no_of_previous_bookings_not_canceled": [0, 2],
                "avg_price_per_room": [65.0, 106.68],
                "no_of_special_requests": [0, 1],
                "booking_status": ["Not_Canceled", "Canceled"],
            }
        )
        result = validator.validate(data)
        # Missing values generate warnings, not errors
        assert len(result.warnings) > 0
        assert any("missing" in warning.lower() for warning in result.warnings)

    def test_validation_result_structure(
        self, validator: DataValidator, valid_data: pd.DataFrame
    ):
        """Test ValidationResult has all expected attributes."""
        result = validator.validate(valid_data)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "stats")
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.stats, dict)
