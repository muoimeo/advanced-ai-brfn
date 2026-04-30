from __future__ import annotations

from dataclasses import asdict

from src.quality.profiles import get_quality_profile
from src.quality.schemas import (
    ImageQualityFeatures,
    ModelPrediction,
    QualityDecision,
    TopKPrediction,
)


LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.60
LOW_TOP1_TOP2_MARGIN_THRESHOLD = 0.15
OPPOSITE_CONDITION_CONFLICT_MARGIN = 0.15
VERY_POOR_IMAGE_QUALITY_THRESHOLD = 15.0
IMAGE_QUALITY_WARNING_THRESHOLD = 35.0

DEFAULT_WEIGHTS = {
    "model_condition": 0.45,
    "color": 0.25,
    "defect_absence": 0.15,
    "image_quality": 0.10,
    "size_proxy": 0.05,
}

WEIGHT_SETS = {
    "default_balanced": DEFAULT_WEIGHTS,
    "more_visual_colour": {
        "model_condition": 0.35,
        "color": 0.30,
        "defect_absence": 0.20,
        "image_quality": 0.10,
        "size_proxy": 0.05,
    },
    "safer_model_led": {
        "model_condition": 0.55,
        "color": 0.20,
        "defect_absence": 0.10,
        "image_quality": 0.10,
        "size_proxy": 0.05,
    },
}

ACTION_BY_GRADE = {
    "A": {
        "action": "normal_sale",
        "inventory_status": "available",
        "discount_percentage": 0,
    },
    "B": {
        "action": "discount_or_quick_sale",
        "inventory_status": "surplus_or_lower_grade",
        "discount_percentage": 20,
    },
    "C": {
        "action": "manual_review_or_discard",
        "inventory_status": "blocked_pending_review",
        "discount_percentage": None,
    },
    "Review": {
        "action": "manual_review_required",
        "inventory_status": "pending_review",
        "discount_percentage": None,
    },
}


def quality_decision_to_dict(decision: QualityDecision) -> dict:
    return asdict(decision)


def model_prediction_to_dict(prediction: ModelPrediction) -> dict:
    data = asdict(prediction)
    data["top_k"] = [asdict(item) for item in prediction.top_k]
    return data


def image_features_to_dict(features: ImageQualityFeatures) -> dict:
    return asdict(features)


def model_condition_score(prediction: ModelPrediction) -> float:
    if prediction.condition == "healthy":
        return prediction.confidence * 100.0
    if prediction.condition == "rotten":
        return (1.0 - prediction.confidence) * 100.0
    return 0.0


def _same_produce_opposite_condition_probability(
    prediction: ModelPrediction,
) -> float | None:
    opposite = "rotten" if prediction.condition == "healthy" else "healthy"
    prefix = f"{prediction.product_type}__".lower()
    for item in prediction.top_k:
        class_name = item.class_name.lower()
        if class_name.startswith(prefix) and class_name.endswith(opposite):
            return item.probability
    return None


def has_opposite_condition_conflict(prediction: ModelPrediction) -> bool:
    opposite_probability = _same_produce_opposite_condition_probability(prediction)
    if opposite_probability is None:
        return False
    return prediction.confidence - opposite_probability < OPPOSITE_CONDITION_CONFLICT_MARGIN


def build_component_scores(
    prediction: ModelPrediction,
    features: ImageQualityFeatures,
) -> dict[str, float]:
    condition_score = model_condition_score(prediction)
    defect_absence_score = features.defect_absence_score
    if prediction.condition == "rotten":
        defect_absence_score = min(defect_absence_score, condition_score)

    ripeness_score = (
        0.50 * condition_score
        + 0.35 * features.color_score
        + 0.15 * defect_absence_score
    )

    return {
        "model_condition": round(condition_score, 4),
        "color": round(features.color_score, 4),
        "defect_absence": round(defect_absence_score, 4),
        "image_quality": round(features.image_quality_score, 4),
        "size_proxy": round(features.size_proxy_score, 4),
        "ripeness": round(ripeness_score, 4),
    }


def calculate_overall_quality_score(
    component_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    selected_weights = weights or DEFAULT_WEIGHTS
    return round(
        sum(component_scores[key] * weight for key, weight in selected_weights.items()),
        4,
    )


def _action_for_grade(grade: str) -> dict:
    return ACTION_BY_GRADE[grade]


def build_quality_decision(
    prediction: ModelPrediction,
    features: ImageQualityFeatures,
    weights: dict[str, float] | None = None,
) -> QualityDecision:
    component_scores = build_component_scores(prediction, features)
    overall_quality_score = calculate_overall_quality_score(component_scores, weights)
    _, profile_warnings = get_quality_profile(prediction.product_type)
    warnings = list(
        dict.fromkeys(
            [
                *profile_warnings,
                *features.feature_warnings,
                "quality_grade_is_rule_based_not_supervised_model_output",
            ]
        )
    )
    reason_codes: list[str] = []
    manual_review = False

    if prediction.condition == "healthy":
        reason_codes.append("classified_as_healthy")
    elif prediction.condition == "rotten":
        reason_codes.append("classified_as_rotten")
    else:
        reason_codes.append("unknown_condition")
        manual_review = True

    if prediction.confidence < LOW_CONFIDENCE_REVIEW_THRESHOLD:
        grade = "Review"
        manual_review = True
        reason_codes.extend(["low_model_confidence", "manual_review_required"])
    elif prediction.condition == "rotten" and prediction.confidence >= 0.80:
        grade = "C"
        reason_codes.append("high_confidence_rotten_prediction")
        if features.image_quality_score < IMAGE_QUALITY_WARNING_THRESHOLD:
            manual_review = True
            reason_codes.extend(["image_quality_warning", "manual_review_required"])
    elif prediction.condition == "rotten":
        grade = "C"
        manual_review = True
        reason_codes.extend(
            ["possible_rotten_prediction_manual_review", "manual_review_required"]
        )
        if features.image_quality_score < IMAGE_QUALITY_WARNING_THRESHOLD:
            reason_codes.append("image_quality_warning")
    elif features.image_quality_score < VERY_POOR_IMAGE_QUALITY_THRESHOLD:
        grade = "Review"
        manual_review = True
        reason_codes.extend(["very_poor_image_quality", "manual_review_required"])
    elif (
        features.dark_ratio >= 0.32
        or component_scores["ripeness"] < 60.0
        or features.color_score < 65.0
        or overall_quality_score < 65.0
    ):
        grade = "C"
        reason_codes.append("one_or_more_grade_c_thresholds_triggered")
    elif (
        component_scores["model_condition"] >= 85.0
        and features.color_score >= 85.0
        and features.size_proxy_score >= 80.0
        and component_scores["ripeness"] >= 80.0
        and overall_quality_score >= 85.0
        and not any(warning.startswith("critical_") for warning in warnings)
    ):
        grade = "A"
        reason_codes.append("all_grade_a_thresholds_met")
    else:
        grade = "B"
        reason_codes.append("acceptable_quality_but_not_grade_a")

    if (
        prediction.condition == "healthy"
        and features.image_quality_score < IMAGE_QUALITY_WARNING_THRESHOLD
        and "very_poor_image_quality" not in reason_codes
    ):
        manual_review = True
        reason_codes.extend(["image_quality_warning", "manual_review_recommended"])

    if (
        prediction.top1_top2_margin is not None
        and prediction.top1_top2_margin < LOW_TOP1_TOP2_MARGIN_THRESHOLD
    ):
        manual_review = True
        reason_codes.extend(["low_top1_top2_margin", "manual_review_required"])

    if has_opposite_condition_conflict(prediction):
        manual_review = True
        reason_codes.extend(["opposite_condition_conflict", "manual_review_required"])

    if "unknown_produce_profile_used" in warnings:
        reason_codes.append("default_quality_profile_used")

    action = _action_for_grade(grade).copy()
    if grade == "C" and manual_review:
        action["action"] = "manual_review_before_listing"

    return QualityDecision(
        grade=grade,
        overall_quality_score=overall_quality_score,
        component_scores=component_scores,
        action=action["action"],
        inventory_status=action["inventory_status"],
        discount_percentage=action["discount_percentage"],
        manual_review=manual_review,
        reason_codes=list(dict.fromkeys(reason_codes)),
        warnings=warnings,
    )


def top_k_from_dicts(top_predictions: list[dict]) -> list[TopKPrediction]:
    top_k = []
    for item in top_predictions:
        class_name = item.get("class_name") or item.get("class")
        probability = item.get("confidence", item.get("probability"))
        if class_name is None or probability is None:
            continue
        top_k.append(TopKPrediction(class_name=class_name, probability=float(probability)))
    return top_k
