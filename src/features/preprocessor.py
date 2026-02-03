"""
Preprocessing Module
====================

Handles data preprocessing using scikit-learn pipelines.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Handles preprocessing of features for the ML model.

    Uses sklearn ColumnTransformer for consistent preprocessing
    of numerical and categorical features.
    """

    # Define feature types
    NUMERICAL_FEATURES = [
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
        "total_nights",
        "total_guests",
    ]

    CATEGORICAL_FEATURES = [
        "type_of_meal_plan",
        "room_type_reserved",
        "market_segment_type",
    ]

    BINARY_FEATURES = [
        "required_car_parking_space",
        "repeated_guest",
    ]

    TARGET_COLUMN = "booking_status"

    def __init__(self):
        """Initialize the Preprocessor."""
        self._pipeline: Optional[ColumnTransformer] = None
        self._is_fitted: bool = False
        self._feature_names_out: Optional[List[str]] = None

    @property
    def is_fitted(self) -> bool:
        """Check if the preprocessor has been fitted."""
        return self._is_fitted

    @property
    def pipeline(self) -> Optional[ColumnTransformer]:
        """Return the sklearn pipeline."""
        return self._pipeline

    def _get_available_features(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Get features that exist in the DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (numerical, categorical, binary) feature lists.
        """
        num_features = [f for f in self.NUMERICAL_FEATURES if f in df.columns]
        cat_features = [f for f in self.CATEGORICAL_FEATURES if f in df.columns]
        bin_features = [f for f in self.BINARY_FEATURES if f in df.columns]

        return num_features, cat_features, bin_features

    def _build_pipeline(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        binary_features: List[str],
    ) -> ColumnTransformer:
        """
        Build the preprocessing pipeline.

        Args:
            numerical_features: List of numerical column names.
            categorical_features: List of categorical column names.
            binary_features: List of binary column names.

        Returns:
            Configured ColumnTransformer.
        """
        # Numerical pipeline: impute missing with median, then scale
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Categorical pipeline: impute missing with most frequent, then one-hot encode
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        # Binary pipeline: just impute missing with most frequent (already 0/1)
        binary_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        # Combine all transformers
        transformers = []

        if numerical_features:
            transformers.append(("numerical", numerical_pipeline, numerical_features))

        if categorical_features:
            transformers.append(
                ("categorical", categorical_pipeline, categorical_features)
            )

        if binary_features:
            transformers.append(("binary", binary_pipeline, binary_features))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop any columns not specified
        )

        return preprocessor

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            df: Training DataFrame (should not include target).

        Returns:
            Self for method chaining.
        """
        num_feat, cat_feat, bin_feat = self._get_available_features(df)

        logger.info(
            f"Fitting preprocessor with {len(num_feat)} numerical, "
            f"{len(cat_feat)} categorical, {len(bin_feat)} binary features"
        )

        self._pipeline = self._build_pipeline(num_feat, cat_feat, bin_feat)
        self._pipeline.fit(df)
        self._is_fitted = True

        # Store feature names
        self._feature_names_out = self._get_feature_names(num_feat, cat_feat, bin_feat)

        logger.info(
            f"Preprocessor fitted. Output features: {len(self._feature_names_out)}"
        )

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted preprocessor.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed numpy array.

        Raises:
            ValueError: If preprocessor hasn't been fitted.
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform.")

        return self._pipeline.transform(df)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform.

        Returns:
            Transformed numpy array.
        """
        self.fit(df)
        return self.transform(df)

    def _get_feature_names(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        binary_features: List[str],
    ) -> List[str]:
        """
        Get output feature names after transformation.

        Args:
            numerical_features: Original numerical feature names.
            categorical_features: Original categorical feature names.
            binary_features: Original binary feature names.

        Returns:
            List of output feature names.
        """
        feature_names = []

        # Numerical features keep their names (after scaling)
        feature_names.extend(numerical_features)

        # Categorical features get one-hot encoded names
        if categorical_features and self._pipeline is not None:
            try:
                cat_transformer = self._pipeline.named_transformers_["categorical"]
                encoder = cat_transformer.named_steps["encoder"]
                cat_feature_names = encoder.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names.tolist())
            except (KeyError, AttributeError):
                # Fallback if encoder doesn't have get_feature_names_out
                feature_names.extend([f"{col}_encoded" for col in categorical_features])

        # Binary features keep their names
        feature_names.extend(binary_features)

        return feature_names

    def get_feature_names_out(self) -> List[str]:
        """
        Get the output feature names.

        Returns:
            List of feature names.

        Raises:
            ValueError: If preprocessor hasn't been fitted.
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted first.")

        return self._feature_names_out or []
