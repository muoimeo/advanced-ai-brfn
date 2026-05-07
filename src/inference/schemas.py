from __future__ import annotations

from datetime import datetime
from typing import Literal
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


class MonitoringDailyRow(BaseModel):
    date: str
    feedback_count: int
    accepted_count: int = 0
    override_count: int = 0
    class_labelled_count: int = 0
    class_correct_count: int = 0
    grade_labelled_count: int = 0
    grade_correct_count: int = 0
    high_confidence_override_count: int = 0
    manual_review_feedback_count: int = 0
    class_accuracy_proxy: float | None = None
    grade_accuracy_proxy: float | None = None
    acceptance_rate: float | None = None
    override_rate: float | None = None


class MonitoringBreakdownRow(BaseModel):
    labelled_count: int
    correct_count: int
    override_count: int
    class_accuracy_proxy: float | None = None


class MonitoringClassRow(MonitoringBreakdownRow):
    predicted_class: str


class MonitoringModelRow(MonitoringBreakdownRow):
    model_name: str


class MonitoringSummaryResponse(BaseModel):
    monitoring_type: str
    limitations: list[str] = Field(default_factory=list)
    prediction_log_count: int
    feedback_log_count: int
    labelled_class_feedback_count: int
    class_accuracy_proxy: float | None = None
    labelled_grade_feedback_count: int
    grade_accuracy_proxy: float | None = None
    accepted_feedback_count: int
    override_feedback_count: int
    override_rate: float | None = None
    high_confidence_override_count: int
    daily: list[MonitoringDailyRow] = Field(default_factory=list)
    by_predicted_class: list[MonitoringClassRow] = Field(default_factory=list)
    by_model: list[MonitoringModelRow] = Field(default_factory=list)


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


class DiscoveryRecommendationResponse(BaseModel):
    customer_id: str
    customer_type: str | None = None
    method: str
    rank: int
    product_id: str
    product_name: str
    producer_id: str
    score: float
    score_components: dict[str, float] = Field(default_factory=dict)
    based_on_product_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    reason_text: str | None = None


class ReorderResponse(BaseModel):
    customer_id: str
    recommendation_date: str
    method: str
    top_k: int
    recommendations: list[ReorderRecommendationResponse] = Field(default_factory=list)
    quick_reorder: list[ReorderRecommendationResponse] | None = None
    you_may_also_like: list[DiscoveryRecommendationResponse] | None = None
    limitations: list[str] = Field(default_factory=list)


class ProducerForecastItemResponse(BaseModel):
    product_id: str
    product_name: str
    forecast_week_start: str
    predicted_quantity_next_week: float
    trend_direction: str
    basis: str
    alert_text: str


class ProducerForecastResponse(BaseModel):
    producer_id: str
    forecast_method: str
    top_k: int
    items: list[ProducerForecastItemResponse] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class RecommenderOrderItemIngest(BaseModel):
    product_id: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)


class RecommenderOrderIngestRequest(BaseModel):
    order_id: str
    customer_id: str
    order_date: str
    items: list[RecommenderOrderItemIngest] = Field(min_items=1)
    total_amount: float | None = None
    customer_type: str | None = None
    postcode_area: str | None = None
    source: str = "desd_order_event"


class RecommenderOrderIngestResponse(BaseModel):
    status: str
    order_id: str
    customer_id: str
    ingested_order_lines: int
    total_orders: int
    total_order_lines: int
    event_log: str
    limitations: list[str] = Field(default_factory=list)


class RecommenderHistoryOrderIngest(BaseModel):
    order_id: str
    order_date: str
    items: list[RecommenderOrderItemIngest] = Field(min_items=1)
    total_amount: float | None = None


class RecommenderHistoryIngestRequest(BaseModel):
    customer_id: str
    orders: list[RecommenderHistoryOrderIngest] = Field(min_items=1)
    customer_type: str | None = None
    postcode_area: str | None = None
    source: str = "desd_history_backfill"


class RecommenderHistoryIngestResponse(BaseModel):
    status: str
    history_batch_id: str
    customer_id: str
    ingested_orders: int
    ingested_order_lines: int
    total_orders: int
    total_order_lines: int
    event_log: str
    limitations: list[str] = Field(default_factory=list)


class CatalogProducerIngestRequest(BaseModel):
    event_type: Literal["producer_upserted"]
    producer_id: str
    producer_name: str
    postcode_area: str
    categories: list[str] = Field(default_factory=list)
    organic_certified: bool | None = None
    created_at: datetime
    source: str


class CatalogProducerIngestResponse(BaseModel):
    status: str
    producer_id: str
    catalog_producers: int
    event_log: str
    limitations: list[str] = Field(default_factory=list)


class CatalogProductIngestRequest(BaseModel):
    event_type: Literal["product_upserted"]
    product_id: str
    producer_id: str
    product_name: str
    category: str
    unit: str
    price: float = Field(ge=0)
    seasonal: bool
    seasonal_start_month: int | None = None
    seasonal_end_month: int | None = None
    available: bool
    created_at: datetime
    source: str


class CatalogProductIngestResponse(BaseModel):
    status: str
    product_id: str
    producer_id: str
    available: bool
    catalog_products: int
    event_log: str
    limitations: list[str] = Field(default_factory=list)
