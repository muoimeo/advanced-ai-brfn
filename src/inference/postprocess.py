from __future__ import annotations


LOW_CONFIDENCE_THRESHOLD = 0.60
AMBIGUOUS_MARGIN_THRESHOLD = 0.15
HIGH_CONFIDENCE_THRESHOLD = 0.95


def split_class_name(class_name: str) -> tuple[str, str]:
    if "__" not in class_name:
        return class_name, "unknown"

    produce_type, status = class_name.split("__", maxsplit=1)
    return produce_type, status.lower()


def quality_grade_from_prediction(freshness_status: str, confidence: float) -> str:
    if freshness_status == "healthy":
        if confidence >= 0.90:
            return "A"
        if confidence >= LOW_CONFIDENCE_THRESHOLD:
            return "B"
        return "Review"

    if freshness_status == "rotten":
        if confidence >= 0.90:
            return "Reject"
        if confidence >= LOW_CONFIDENCE_THRESHOLD:
            return "Likely reject"
        return "Review"

    return "Review"


def recommended_action_from_prediction(
    freshness_status: str,
    confidence: float,
    top1_top2_margin: float | None = None,
) -> str:
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "Manual inspection recommended before taking action."

    if (
        top1_top2_margin is not None
        and top1_top2_margin < AMBIGUOUS_MARGIN_THRESHOLD
    ):
        return "Manual inspection recommended before taking action."

    if freshness_status == "healthy":
        return "Accept for marketplace listing or normal handling."

    if freshness_status == "rotten":
        return "Reject from sale and route for disposal or supplier review."

    return "Manual inspection recommended before taking action."


def reason_codes_from_prediction(
    freshness_status: str,
    confidence: float,
    top1_top2_margin: float | None = None,
) -> list[str]:
    reason_codes = []

    if freshness_status == "rotten":
        reason_codes.append("ROTTEN_PREDICTION")
    elif freshness_status == "healthy":
        reason_codes.append("HEALTHY_PREDICTION")
    else:
        reason_codes.append("UNKNOWN_FRESHNESS_STATUS")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        reason_codes.extend(["LOW_CONFIDENCE", "MANUAL_REVIEW_REQUIRED"])
    elif confidence < 0.90:
        reason_codes.append("MEDIUM_CONFIDENCE")
    else:
        reason_codes.append("HIGH_CONFIDENCE")

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        reason_codes.append("HIGH_CONFIDENCE_PREDICTION")

    if (
        top1_top2_margin is not None
        and top1_top2_margin < AMBIGUOUS_MARGIN_THRESHOLD
    ):
        reason_codes.extend(["AMBIGUOUS_PREDICTION", "MANUAL_REVIEW_REQUIRED"])

    if freshness_status == "rotten" and confidence >= 0.90:
        reason_codes.append("HIGH_CONFIDENCE_REJECTION")

    return list(dict.fromkeys(reason_codes))


def manual_review_required_from_prediction(
    freshness_status: str,
    confidence: float,
    top1_top2_margin: float | None = None,
) -> bool:
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return True

    if freshness_status not in {"healthy", "rotten"}:
        return True

    return (
        top1_top2_margin is not None
        and top1_top2_margin < AMBIGUOUS_MARGIN_THRESHOLD
    )


def freshness_score_from_prediction(freshness_status: str, confidence: float) -> float:
    if freshness_status == "healthy":
        return confidence
    if freshness_status == "rotten":
        return 1.0 - confidence
    return 0.0


def build_business_prediction(
    predicted_class: str,
    confidence: float,
    top_predictions: list[dict] | None = None,
    top1_top2_margin: float | None = None,
) -> dict:
    produce_type, freshness_status = split_class_name(predicted_class)
    if top1_top2_margin is None and top_predictions and len(top_predictions) >= 2:
        top1_top2_margin = (
            float(top_predictions[0]["confidence"])
            - float(top_predictions[1]["confidence"])
        )

    reason_codes = reason_codes_from_prediction(
        freshness_status,
        confidence,
        top1_top2_margin,
    )

    return {
        "produce_type": produce_type,
        "predicted_class": predicted_class,
        "freshness_status": freshness_status,
        "confidence": confidence,
        "top1_top2_margin": top1_top2_margin,
        "confidence_score": confidence,
        "freshness_score": freshness_score_from_prediction(freshness_status, confidence),
        "quality_grade": quality_grade_from_prediction(freshness_status, confidence),
        "recommended_action": recommended_action_from_prediction(
            freshness_status,
            confidence,
            top1_top2_margin,
        ),
        "reason_codes": reason_codes,
        "manual_review_required": manual_review_required_from_prediction(
            freshness_status,
            confidence,
            top1_top2_margin,
        ),
        "top_predictions": top_predictions or [],
    }
