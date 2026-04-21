from __future__ import annotations


def split_class_name(class_name: str) -> tuple[str, str]:
    if "__" not in class_name:
        return class_name, "unknown"

    produce_type, status = class_name.split("__", maxsplit=1)
    return produce_type, status.lower()


def quality_grade_from_prediction(freshness_status: str, confidence: float) -> str:
    if freshness_status == "healthy":
        if confidence >= 0.90:
            return "A"
        if confidence >= 0.70:
            return "B"
        return "Review"

    if freshness_status == "rotten":
        if confidence >= 0.90:
            return "Reject"
        if confidence >= 0.70:
            return "Likely reject"
        return "Review"

    return "Review"


def recommended_action_from_prediction(freshness_status: str, confidence: float) -> str:
    if confidence < 0.70:
        return "Manual inspection recommended before taking action."

    if freshness_status == "healthy":
        return "Accept for marketplace listing or normal handling."

    if freshness_status == "rotten":
        return "Reject from sale and route for disposal or supplier review."

    return "Manual inspection recommended before taking action."


def build_business_prediction(
    predicted_class: str,
    confidence: float,
    top_predictions: list[dict] | None = None,
) -> dict:
    produce_type, freshness_status = split_class_name(predicted_class)

    return {
        "produce_type": produce_type,
        "predicted_class": predicted_class,
        "freshness_status": freshness_status,
        "confidence": confidence,
        "quality_grade": quality_grade_from_prediction(freshness_status, confidence),
        "recommended_action": recommended_action_from_prediction(freshness_status, confidence),
        "top_predictions": top_predictions or [],
    }
