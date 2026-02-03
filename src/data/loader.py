"""
Data Loading Module
===================

Handles loading and initial validation of hotel reservation data.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Responsible for loading hotel reservation data from various sources.

    Follows Single Responsibility Principle - only handles data loading.
    """

    EXPECTED_COLUMNS = [
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

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        """
        Initialize the DataLoader.

        Args:
            data_path: Path to the data file. If None, uses default path.
        """
        self.data_path = Path(data_path) if data_path else None

    def load_csv(self, path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            path: Path to CSV file. Uses instance path if not provided.

        Returns:
            DataFrame with loaded data.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If required columns are missing.
        """
        file_path = Path(path) if path else self.data_path

        if file_path is None:
            raise ValueError("No data path provided.")

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")

        df = pd.read_csv(file_path)

        self._validate_columns(df)

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that all expected columns are present.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If required columns are missing.
        """
        missing_columns = set(self.EXPECTED_COLUMNS) - set(df.columns)

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def load_from_dict(self, data: dict) -> pd.DataFrame:
        """
        Load data from a dictionary (useful for API requests).

        Args:
            data: Dictionary with column names as keys.

        Returns:
            DataFrame with loaded data.
        """
        df = pd.DataFrame(data)
        return df
