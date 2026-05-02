from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_name: str
    confidence: float


class PredictionEvidence(BaseModel):
    predicted_class: str
    product_type: str
    condition: str
    confidence: float
    top_k: list[dict[str, Any]] = Field(default_factory=list)
    top1_top2_margin: float | None = None


class QualityDecisionResponse(BaseModel):
    grade: str
    overall_quality_score: float
    component_scores: dict[str, float] = Field(default_factory=dict)
    action: str
    inventory_status: str
    discount_percentage: int | None = None
    manual_review: bool
    reason_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class XAIReference(BaseModel):
    method: str | None = None
    available: bool = False
    heatmap_path: str | None = None
    note: str | None = None


class PredictionResponse(BaseModel):
    prediction_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    produce_type: str
    predicted_class: str
    freshness_status: str
    confidence: float
    top1_top2_margin: float | None = None
    confidence_score: float
    freshness_score: float
    quality_grade: str
    recommended_action: str
    reason_codes: list[str] = Field(default_factory=list)
    manual_review_required: bool
    top_predictions: list[TopPrediction] = Field(default_factory=list)
    prediction: PredictionEvidence | None = None
    quality: QualityDecisionResponse | None = None
    xai: XAIReference | None = None
    model_info: dict[str, Any] | None = None
    image_features: dict[str, Any] | None = None


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
    prediction_id: str | None = None
    image_id: str | None = None
    predicted_class: str | None = None
    predicted_grade: str | None = None
    corrected_class: str | None = None
    producer_override_class: str | None = None
    producer_override_grade: str | None = None
    override_reason: str | None = None
    accepted_ai_recommendation: bool | None = None
    quality_decision_snapshot: dict[str, Any] | None = None
    model_name: str | None = None
    user_note: str | None = None


class FeedbackResponse(BaseModel):
    status: str


class ReorderRecommendationResponse(BaseModel):
    customer_id: str
    customer_type: str | None = None
    method: str
    rank: int
    product_id: str
    product_name: str
    producer_id: str
    score: float
    reason_codes: list[str] = Field(default_factory=list)
    reason_text: str | None = None


class ReorderResponse(BaseModel):
    customer_id: str
    recommendation_date: str
    method: str
    top_k: int
    recommendations: list[ReorderRecommendationResponse] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
