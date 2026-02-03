"""
API Schemas
===========

Pydantic models for request/response validation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class ReservationInput(BaseModel):
    """Input schema for a single hotel reservation."""

    no_of_adults: int = Field(..., ge=0, description="Number of adults")
    no_of_children: int = Field(default=0, ge=0, description="Number of children")
    no_of_weekend_nights: int = Field(..., ge=0, description="Number of weekend nights")
    no_of_week_nights: int = Field(..., ge=0, description="Number of weekday nights")
    type_of_meal_plan: str = Field(..., description="Type of meal plan")
    required_car_parking_space: int = Field(
        default=0, ge=0, le=1, description="Car parking requested (0/1)"
    )
    room_type_reserved: str = Field(..., description="Room type reserved")
    lead_time: int = Field(..., ge=0, description="Days between booking and arrival")
    arrival_year: int = Field(..., ge=2000, description="Year of arrival")
    arrival_month: int = Field(..., ge=1, le=12, description="Month of arrival")
    arrival_date: int = Field(..., ge=1, le=31, description="Day of arrival")
    market_segment_type: str = Field(..., description="Market segment type")
    repeated_guest: int = Field(
        default=0, ge=0, le=1, description="Is repeated guest (0/1)"
    )
    no_of_previous_cancellations: int = Field(
        default=0, ge=0, description="Number of previous cancellations"
    )
    no_of_previous_bookings_not_canceled: int = Field(
        default=0, ge=0, description="Number of previous bookings not canceled"
    )
    avg_price_per_room: float = Field(..., ge=0, description="Average price per room")
    no_of_special_requests: int = Field(
        default=0, ge=0, description="Number of special requests"
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response schema for a single prediction."""

    prediction: int = Field(..., description="Prediction (0=Not Canceled, 1=Canceled)")
    probability: float = Field(..., description="Cancellation probability")
    label: str = Field(..., description="Human-readable label")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    reservations: List[ReservationInput] = Field(
        ..., min_length=1, description="List of reservations"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
