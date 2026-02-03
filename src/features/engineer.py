"""
Feature Engineering Module
==========================

Handles creation of derived features for the hotel reservation model.
"""

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates derived features from raw data.

    Implements feature engineering strategies specific to hotel reservation
    cancellation prediction.
    """

    # Columns to drop (high cardinality, data leakage risk)
    COLUMNS_TO_DROP = ["Booking_ID"]

    # Target column
    TARGET_COLUMN = "booking_status"
    TARGET_MAPPING = {"Canceled": 1, "Not_Canceled": 0}

    def __init__(self):
        """Initialize the FeatureEngineer."""
        self._feature_names: Optional[List[str]] = None

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Return the list of feature names after transformation."""
        return self._feature_names

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the data with feature engineering.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame with engineered features.
        """
        df = df.copy()
        df = self._drop_columns(df)
        df = self._create_total_nights(df)
        df = self._create_total_guests(df)
        df = self._encode_target(df)

        self._feature_names = [col for col in df.columns if col != self.TARGET_COLUMN]

        feature_count = len(self._feature_names)
        logger.info(f"Feature engineering complete. {feature_count} features created.")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with feature engineering (for inference).

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        df = df.copy()
        df = self._drop_columns(df, ignore_missing=True)
        df = self._create_total_nights(df)
        df = self._create_total_guests(df)

        # Don't encode target for inference (might not exist)
        if self.TARGET_COLUMN in df.columns:
            df = self._encode_target(df)

        return df

    def _drop_columns(
        self, df: pd.DataFrame, ignore_missing: bool = False
    ) -> pd.DataFrame:
        """
        Drop columns that shouldn't be used for modeling.

        Args:
            df: Input DataFrame.
            ignore_missing: If True, don't raise error for missing columns.

        Returns:
            DataFrame without dropped columns.
        """
        cols_to_drop = [col for col in self.COLUMNS_TO_DROP if col in df.columns]

        if cols_to_drop:
            logger.info(f"Dropping columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        return df

    def _create_total_nights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create total_nights feature.

        Combines weekend and weekday nights into a single feature.
        """
        if "no_of_weekend_nights" in df.columns and "no_of_week_nights" in df.columns:
            df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
            logger.debug("Created 'total_nights' feature")

        return df

    def _create_total_guests(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create total_guests feature.

        Combines adults and children into total guest count.
        """
        if "no_of_adults" in df.columns and "no_of_children" in df.columns:
            df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
            logger.debug("Created 'total_guests' feature")

        return df

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the target variable.

        Maps 'Canceled' -> 1, 'Not_Canceled' -> 0
        """
        if self.TARGET_COLUMN in df.columns:
            df[self.TARGET_COLUMN] = df[self.TARGET_COLUMN].map(self.TARGET_MAPPING)
            logger.info(f"Encoded target column: {self.TARGET_MAPPING}")

        return df

    def get_feature_importance_names(self) -> List[str]:
        """
        Return feature names for feature importance analysis.

        Returns:
            List of feature names.
        """
        if self._feature_names is None:
            raise ValueError("FeatureEngineer has not been fitted yet.")
        return self._feature_names
