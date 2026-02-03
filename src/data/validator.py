"""
Data Validation Module
======================

Handles data quality checks and validation.
"""

import logging
from dataclasses import dataclass
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: dict


class DataValidator:
    """
    Validates data quality and integrity.

    Performs checks for:
    - Missing values
    - Data types
    - Value ranges
    - Categorical values
    """

    CATEGORICAL_COLUMNS = [
        "type_of_meal_plan",
        "room_type_reserved",
        "market_segment_type",
        "booking_status",
    ]

    NUMERIC_COLUMNS = [
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "lead_time",
        "arrival_year",
        "arrival_month",
        "arrival_date",
        "avg_price_per_room",
        "no_of_special_requests",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
    ]

    BINARY_COLUMNS = ["required_car_parking_space", "repeated_guest"]

    VALID_BOOKING_STATUS = ["Canceled", "Not_Canceled"]
    VALID_MEAL_PLANS = ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
    VALID_MARKET_SEGMENTS = [
        "Online",
        "Offline",
        "Corporate",
        "Aviation",
        "Complementary",
    ]

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Perform comprehensive data validation.

        Args:
            df: DataFrame to validate.

        Returns:
            ValidationResult with validation outcome.
        """
        errors = []
        warnings = []

        # Check for missing values
        missing_check = self._check_missing_values(df)
        warnings.extend(missing_check)

        # Check data types
        type_errors = self._check_data_types(df)
        errors.extend(type_errors)

        # Check value ranges
        range_errors = self._check_value_ranges(df)
        errors.extend(range_errors)

        # Check categorical values
        cat_errors = self._check_categorical_values(df)
        errors.extend(cat_errors)

        # Compute statistics
        stats = self._compute_stats(df)

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation failed with {len(errors)} errors")

        return ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, stats=stats
        )

    def _check_missing_values(self, df: pd.DataFrame) -> List[str]:
        """Check for missing values in the DataFrame."""
        warnings = []
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        for col, count in missing.items():
            pct = (count / len(df)) * 100
            warnings.append(f"Column '{col}' has {count} missing values ({pct:.2f}%)")

        return warnings

    def _check_data_types(self, df: pd.DataFrame) -> List[str]:
        """Verify data types are as expected."""
        errors = []

        for col in self.NUMERIC_COLUMNS:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(
                    f"Column '{col}' should be numeric but is {df[col].dtype}"
                )

        return errors

    def _check_value_ranges(self, df: pd.DataFrame) -> List[str]:
        """Check that numeric values are within expected ranges."""
        errors = []

        # Check non-negative columns
        non_negative_cols = [
            "no_of_adults",
            "no_of_children",
            "lead_time",
            "no_of_weekend_nights",
            "no_of_week_nights",
        ]

        for col in non_negative_cols:
            if col in df.columns:
                if (df[col] < 0).any():
                    errors.append(f"Column '{col}' contains negative values")

        # Check month range
        if "arrival_month" in df.columns:
            if not df["arrival_month"].between(1, 12).all():
                errors.append("arrival_month contains values outside 1-12 range")

        # Check date range
        if "arrival_date" in df.columns:
            if not df["arrival_date"].between(1, 31).all():
                errors.append("arrival_date contains values outside 1-31 range")

        return errors

    def _check_categorical_values(self, df: pd.DataFrame) -> List[str]:
        """Verify categorical values are from expected sets."""
        errors = []

        if "booking_status" in df.columns:
            invalid = set(df["booking_status"].unique()) - set(
                self.VALID_BOOKING_STATUS
            )
            if invalid:
                errors.append(f"Invalid booking_status values: {invalid}")

        return errors

    def _compute_stats(self, df: pd.DataFrame) -> dict:
        """Compute summary statistics for the dataset."""
        stats = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "missing_values_total": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
        }

        if "booking_status" in df.columns:
            value_counts = df["booking_status"].value_counts()
            stats["target_distribution"] = value_counts.to_dict()
            if "Canceled" in value_counts.index:
                stats["cancellation_rate"] = (
                    value_counts.get("Canceled", 0) / len(df) * 100
                )

        return stats
