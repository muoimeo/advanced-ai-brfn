from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TopKPrediction:
    class_name: str
    probability: float


@dataclass(frozen=True)
class ModelPrediction:
    predicted_class: str
    product_type: str
    condition: str
    confidence: float
    top_k: list[TopKPrediction] = field(default_factory=list)
    top1_top2_margin: float | None = None


@dataclass(frozen=True)
class ImageQualityFeatures:
    image_quality_score: float
    color_score: float
    defect_absence_score: float
    size_proxy_score: float
    dark_ratio: float
    accepted_color_ratio: float
    blur_score: float
    brightness_score: float
    foreground_area_ratio: float
    feature_warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QualityDecision:
    grade: str
    overall_quality_score: float
    component_scores: dict[str, float]
    action: str
    inventory_status: str
    discount_percentage: int | None
    manual_review: bool
    reason_codes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

