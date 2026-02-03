"""
Model Prediction Module
=======================

Handles model loading and inference.
"""

import logging
from typing import List, Optional, Union

import mlflow
import numpy as np
import pandas as pd

from src.features.engineer import FeatureEngineer
from src.features.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Handles model loading and prediction.

    Responsible for:
    - Loading models from MLflow or local storage
    - Running inference on new data
    - Returning predictions and probabilities
    """

    def __init__(
        self,
        model_uri: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ):
        """
        Initialize the ModelPredictor.

        Args:
            model_uri: MLflow model URI or path to local model.
            preprocessor: Fitted Preprocessor instance.
            feature_engineer: Fitted FeatureEngineer instance.
        """
        self.model_uri = model_uri
        self._model = None
        self._preprocessor = preprocessor
        self._feature_engineer = feature_engineer or FeatureEngineer()

    def load_model(self, model_uri: Optional[str] = None) -> None:
        """
        Load model from MLflow.

        Args:
            model_uri: MLflow model URI. Uses instance URI if not provided.

        Raises:
            ValueError: If no model URI provided.
        """
        uri = model_uri or self.model_uri

        if uri is None:
            raise ValueError("No model URI provided.")

        logger.info(f"Loading model from: {uri}")

        self._model = mlflow.xgboost.load_model(uri)

        logger.info("Model loaded successfully")

    def set_model(self, model) -> None:
        """
        Set the model directly (useful for testing).

        Args:
            model: A trained model instance.
        """
        self._model = model

    def set_preprocessor(self, preprocessor: Preprocessor) -> None:
        """
        Set the preprocessor.

        Args:
            preprocessor: Fitted Preprocessor instance.
        """
        self._preprocessor = preprocessor

    def predict(
        self,
        data: Union[pd.DataFrame, dict, List[dict]],
        return_proba: bool = False,
    ) -> Union[np.ndarray, dict]:
        """
        Make predictions on new data.

        Args:
            data: Input data as DataFrame, dict (single), or list of dicts (batch).
            return_proba: If True, also return prediction probabilities.

        Returns:
            Predictions array, or dict with predictions and probabilities.

        Raises:
            ValueError: If model or preprocessor not loaded.
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self._preprocessor is None:
            raise ValueError("Preprocessor not set. Call set_preprocessor() first.")

        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Apply feature engineering
        df = self._feature_engineer.transform(df)

        # Preprocess
        X = self._preprocessor.transform(df)

        # Predict
        predictions = self._model.predict(X)

        if return_proba:
            probabilities = self._model.predict_proba(X)[:, 1]
            return {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
                "labels": [
                    "Not Canceled" if p == 0 else "Canceled" for p in predictions
                ],
            }

        return predictions

    def predict_single(self, data: dict, return_proba: bool = True) -> dict:
        """
        Make prediction for a single record.

        Args:
            data: Single record as dictionary.
            return_proba: If True, include probability in result.

        Returns:
            Dictionary with prediction results.
        """
        result = self.predict(data, return_proba=return_proba)

        if isinstance(result, dict):
            return {
                "prediction": result["predictions"][0],
                "probability": result["probabilities"][0],
                "label": result["labels"][0],
            }

        return {"prediction": int(result[0])}

    def predict_batch(self, data: List[dict], return_proba: bool = True) -> List[dict]:
        """
        Make predictions for multiple records.

        Args:
            data: List of records as dictionaries.
            return_proba: If True, include probabilities in results.

        Returns:
            List of dictionaries with prediction results.
        """
        result = self.predict(data, return_proba=return_proba)

        if isinstance(result, dict):
            predictions = []
            for i in range(len(result["predictions"])):
                predictions.append(
                    {
                        "prediction": result["predictions"][i],
                        "probability": result["probabilities"][i],
                        "label": result["labels"][i],
                    }
                )
            return predictions

        return [{"prediction": int(p)} for p in result]
