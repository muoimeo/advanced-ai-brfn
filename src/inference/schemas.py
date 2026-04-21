from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    produce_type: str
    predicted_class: str
    freshness_status: str
    confidence: float
    quality_grade: str
    recommended_action: str
    top_predictions: list[TopPrediction] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str | None = None
    base_model_family: str | None = None
    num_classes: int | None = None
    image_size: list[int] | None = None
    metrics: dict[str, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    image_id: str | None = None
    predicted_class: str | None = None
    corrected_class: str | None = None
    user_note: str | None = None


class FeedbackResponse(BaseModel):
    status: str
