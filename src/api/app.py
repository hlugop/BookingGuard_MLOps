"""
FastAPI Application
===================

Main API application for hotel cancellation prediction.
Uses dependency injection and centralized configuration.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    ReservationInput,
)
from src.config import Settings, get_settings
from src.container import Container, reset_container
from src.exceptions import (
    BaseMLOpsException,
    PipelineNotLoadedError,
    PredictionError,
    ServiceUnavailableError,
)
from src.pipelines.inference import InferencePipeline

logger = logging.getLogger(__name__)


class AppState:
    """
    Application state container.

    Replaces global variables with a proper state object
    that can be attached to the FastAPI app.
    """

    def __init__(self):
        self.container: Optional[Container] = None
        self.inference_pipeline: Optional[InferencePipeline] = None
        self.settings: Optional[Settings] = None

    @property
    def is_ready(self) -> bool:
        """Check if the application is ready to serve requests."""
        return self.inference_pipeline is not None and self.inference_pipeline.is_loaded


def get_app_state(request: Request) -> AppState:
    """
    Dependency to get application state.

    Args:
        request: FastAPI request object.

    Returns:
        AppState instance from app.state.
    """
    return request.app.state.app_state


def get_inference_pipeline(
    app_state: AppState = Depends(get_app_state),
) -> InferencePipeline:
    """
    Dependency to get the inference pipeline.

    Args:
        app_state: Application state.

    Returns:
        InferencePipeline instance.

    Raises:
        HTTPException: If pipeline not loaded (503).
    """
    if not app_state.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later.",
        )
    return app_state.inference_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.

    Loads configuration, container, and model on startup.
    """
    # Initialize app state
    app_state = AppState()
    app.state.app_state = app_state

    # Load settings
    app_state.settings = get_settings()
    logger.info(f"Starting application in {app_state.settings.environment} mode")

    # Create container
    app_state.container = Container.from_settings(settings=app_state.settings)

    # Load inference pipeline
    try:
        app_state.inference_pipeline = app_state.container.inference_pipeline(
            load_artifacts=False
        )
        app_state.inference_pipeline.load()
        logger.info("Model loaded successfully")
        logger.info(f"Model URI: {app_state.inference_pipeline.model_uri}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't fail startup - allow health endpoint to report status
        app_state.inference_pipeline = None

    yield

    # Cleanup
    logger.info("Shutting down application...")
    reset_container()


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        settings: Optional Settings instance for configuration.

    Returns:
        Configured FastAPI application.
    """
    # Use provided settings or get from config
    settings = settings or get_settings()

    app = FastAPI(
        title="Hotel Cancellation Prediction API",
        description=(
            "API for predicting hotel reservation cancellations. "
            "Supports single and batch predictions."
        ),
        version=__version__,
        lifespan=lifespan,
        debug=settings.debug,
        responses={
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
            503: {"model": ErrorResponse, "description": "Service Unavailable"},
        },
    )

    # Configure CORS from settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=settings.api.cors_allow_credentials,
        allow_methods=settings.api.cors_allow_methods,
        allow_headers=settings.api.cors_allow_headers,
    )

    # Exception handlers
    @app.exception_handler(BaseMLOpsException)
    async def mlops_exception_handler(
        request: Request, exc: BaseMLOpsException
    ) -> JSONResponse:
        """Handle custom MLOps exceptions."""
        logger.error(f"MLOps exception: {exc.message}", exc_info=exc.original_error)

        status_code = 500
        if isinstance(exc, ServiceUnavailableError):
            status_code = 503
        elif isinstance(exc, PipelineNotLoadedError):
            status_code = 503
        elif isinstance(exc, PredictionError):
            status_code = 500

        return JSONResponse(
            status_code=status_code,
            content=exc.to_dict(),
        )

    # Routes
    @app.get("/", tags=["Root"])
    async def root(app_state: AppState = Depends(get_app_state)):
        """Root endpoint with API information."""
        return {
            "message": "Hotel Cancellation Prediction API",
            "version": __version__,
            "environment": (
                app_state.settings.environment if app_state.settings else "unknown"
            ),
            "docs": "/docs",
        }

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Check API health",
    )
    async def health_check(app_state: AppState = Depends(get_app_state)):
        """
        Check if the API is healthy and model is loaded.
        """
        return HealthResponse(
            status="healthy" if app_state.is_ready else "degraded",
            model_loaded=app_state.is_ready,
            version=__version__,
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["Predictions"],
        summary="Single prediction",
        responses={
            200: {"description": "Successful prediction"},
            503: {"model": ErrorResponse, "description": "Model not loaded"},
        },
    )
    async def predict(
        reservation: ReservationInput,
        pipeline: InferencePipeline = Depends(get_inference_pipeline),
    ):
        """
        Make a prediction for a single hotel reservation.

        Returns:
            Prediction with probability and label.
        """
        try:
            result = pipeline.predict_single(reservation.model_dump())
            return PredictionResponse(**result)
        except PredictionError as e:
            logger.error(f"Prediction error: {e.message}")
            raise HTTPException(status_code=500, detail=e.message)
        except Exception as e:
            logger.error(f"Unexpected prediction error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}",
            )

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["Predictions"],
        summary="Batch predictions",
        responses={
            200: {"description": "Successful batch prediction"},
            503: {"model": ErrorResponse, "description": "Model not loaded"},
        },
    )
    async def predict_batch(
        request: BatchPredictionRequest,
        pipeline: InferencePipeline = Depends(get_inference_pipeline),
    ):
        """
        Make predictions for multiple hotel reservations.

        Returns:
            List of predictions with probabilities and labels.
        """
        try:
            data = [r.model_dump() for r in request.reservations]
            results = pipeline.predict_batch(data)
            predictions = [PredictionResponse(**r) for r in results]
            return BatchPredictionResponse(
                predictions=predictions,
                count=len(predictions),
            )
        except PredictionError as e:
            logger.error(f"Batch prediction error: {e.message}")
            raise HTTPException(status_code=500, detail=e.message)
        except Exception as e:
            logger.error(f"Unexpected batch prediction error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Batch prediction failed: {str(e)}",
            )

    return app


# Create application instance
app = create_app()
