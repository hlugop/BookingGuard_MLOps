"""
Unit Tests for Feature Engineering
==================================

Tests for the FeatureEngineer class.
"""

import pandas as pd
import pytest

from src.features.engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "Booking_ID": ["INN00001", "INN00002"],
                "no_of_adults": [2, 1],
                "no_of_children": [1, 0],
                "no_of_weekend_nights": [1, 2],
                "no_of_week_nights": [2, 3],
                "type_of_meal_plan": ["Meal Plan 1", "Not Selected"],
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

    @pytest.fixture
    def feature_engineer(self) -> FeatureEngineer:
        """Create a FeatureEngineer instance."""
        return FeatureEngineer()

    def test_fit_transform_drops_booking_id(
        self, feature_engineer: FeatureEngineer, sample_data: pd.DataFrame
    ):
        """Test that Booking_ID is dropped after fit_transform."""
        result = feature_engineer.fit_transform(sample_data)
        assert "Booking_ID" not in result.columns

    def test_fit_transform_creates_total_nights(
        self, feature_engineer: FeatureEngineer, sample_data: pd.DataFrame
    ):
        """Test that total_nights feature is created."""
        result = feature_engineer.fit_transform(sample_data)
        assert "total_nights" in result.columns
        # Check calculation: weekend + week nights
        assert result.loc[0, "total_nights"] == 3  # 1 + 2
        assert result.loc[1, "total_nights"] == 5  # 2 + 3

    def test_fit_transform_creates_total_guests(
        self, feature_engineer: FeatureEngineer, sample_data: pd.DataFrame
    ):
        """Test that total_guests feature is created."""
        result = feature_engineer.fit_transform(sample_data)
        assert "total_guests" in result.columns
        # Check calculation: adults + children
        assert result.loc[0, "total_guests"] == 3  # 2 + 1
        assert result.loc[1, "total_guests"] == 1  # 1 + 0

    def test_fit_transform_encodes_target(
        self, feature_engineer: FeatureEngineer, sample_data: pd.DataFrame
    ):
        """Test that booking_status is encoded correctly."""
        result = feature_engineer.fit_transform(sample_data)
        # Not_Canceled -> 0, Canceled -> 1
        assert result.loc[0, "booking_status"] == 0
        assert result.loc[1, "booking_status"] == 1

    def test_transform_without_target(
        self, feature_engineer: FeatureEngineer, sample_data: pd.DataFrame
    ):
        """Test transform works without target column."""
        # First fit
        feature_engineer.fit_transform(sample_data)

        # Create inference data without target
        inference_data = sample_data.drop(columns=["booking_status"])
        result = feature_engineer.transform(inference_data)

        assert "total_nights" in result.columns
        assert "total_guests" in result.columns
        assert "booking_status" not in result.columns

    def test_feature_names_property(
        self, feature_engineer: FeatureEngineer, sample_data: pd.DataFrame
    ):
        """Test that feature_names property returns correct names after fit."""
        feature_engineer.fit_transform(sample_data)
        feature_names = feature_engineer.feature_names

        assert feature_names is not None
        assert "booking_status" not in feature_names
        assert "total_nights" in feature_names
        assert "total_guests" in feature_names

    def test_feature_names_before_fit_returns_none(
        self, feature_engineer: FeatureEngineer
    ):
        """Test that feature_names is None before fitting."""
        assert feature_engineer.feature_names is None

    def test_get_feature_importance_names_raises_before_fit(
        self, feature_engineer: FeatureEngineer
    ):
        """Test that getting feature importance names raises error before fit."""
        with pytest.raises(ValueError, match="has not been fitted"):
            feature_engineer.get_feature_importance_names()

    def test_original_data_not_modified(
        self, feature_engineer: FeatureEngineer, sample_data: pd.DataFrame
    ):
        """Test that original DataFrame is not modified."""
        original_columns = sample_data.columns.tolist()
        feature_engineer.fit_transform(sample_data)
        assert sample_data.columns.tolist() == original_columns
