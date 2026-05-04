from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import LOGS_DIR, OUTPUTS_DIR


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _date_part(value: str | None) -> str:
    if not value:
        return "unknown"
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return value[:10]


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _class_correct(feedback: dict[str, Any], prediction: dict[str, Any] | None) -> bool | None:
    accepted = feedback.get("accepted_ai_recommendation")
    predicted = _coalesce(feedback.get("predicted_class"), prediction.get("predicted_class") if prediction else None)
    corrected = _coalesce(feedback.get("corrected_class"), feedback.get("producer_override_class"))

    if accepted is True:
        return True
    if corrected and predicted:
        return corrected == predicted
    if accepted is False and predicted:
        return False
    return None


def _grade_correct(feedback: dict[str, Any], prediction: dict[str, Any] | None) -> bool | None:
    accepted = feedback.get("accepted_ai_recommendation")
    predicted = _coalesce(feedback.get("predicted_grade"), prediction.get("quality_grade") if prediction else None)
    corrected = feedback.get("producer_override_grade")

    if accepted is True:
        return True
    if corrected and predicted:
        return corrected == predicted
    if accepted is False and predicted:
        return False
    return None


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def build_feedback_monitoring_summary(logs_dir: Path = LOGS_DIR) -> dict[str, Any]:
    """Build monitoring metrics from prediction and feedback JSONL logs.

    The result is an operational monitoring proxy, not a controlled test-set
    accuracy score. A prediction only becomes monitorable when a human accepts
    or overrides it through the feedback endpoint.
    """

    predictions = _read_jsonl(logs_dir / "predictions.jsonl")
    feedback = _read_jsonl(logs_dir / "feedback.jsonl")
    predictions_by_id = {
        record.get("prediction_id"): record for record in predictions if record.get("prediction_id")
    }

    daily: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "feedback_count": 0,
            "accepted_count": 0,
            "override_count": 0,
            "class_labelled_count": 0,
            "class_correct_count": 0,
            "grade_labelled_count": 0,
            "grade_correct_count": 0,
            "high_confidence_override_count": 0,
            "manual_review_feedback_count": 0,
        }
    )
    by_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"labelled_count": 0, "correct_count": 0, "override_count": 0}
    )
    by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"labelled_count": 0, "correct_count": 0, "override_count": 0}
    )

    labelled_class_total = 0
    class_correct_total = 0
    labelled_grade_total = 0
    grade_correct_total = 0
    accepted_total = 0
    override_total = 0
    high_confidence_override_total = 0

    for item in feedback:
        prediction = predictions_by_id.get(item.get("prediction_id"))
        date_key = _date_part(_coalesce(item.get("created_at"), prediction.get("created_at") if prediction else None))
        bucket = daily[date_key]
        bucket["feedback_count"] += 1

        accepted = item.get("accepted_ai_recommendation")
        if accepted is True:
            accepted_total += 1
            bucket["accepted_count"] += 1
        elif accepted is False or item.get("producer_override_class") or item.get("producer_override_grade") or item.get("corrected_class"):
            override_total += 1
            bucket["override_count"] += 1

        class_result = _class_correct(item, prediction)
        predicted_class = _coalesce(item.get("predicted_class"), prediction.get("predicted_class") if prediction else None, "unknown")
        model_name = _coalesce(item.get("model_name"), prediction.get("model_name") if prediction else None, "unknown")

        if class_result is not None:
            labelled_class_total += 1
            bucket["class_labelled_count"] += 1
            by_class[predicted_class]["labelled_count"] += 1
            by_model[model_name]["labelled_count"] += 1
            if class_result:
                class_correct_total += 1
                bucket["class_correct_count"] += 1
                by_class[predicted_class]["correct_count"] += 1
                by_model[model_name]["correct_count"] += 1
            else:
                by_class[predicted_class]["override_count"] += 1
                by_model[model_name]["override_count"] += 1

        grade_result = _grade_correct(item, prediction)
        if grade_result is not None:
            labelled_grade_total += 1
            bucket["grade_labelled_count"] += 1
            if grade_result:
                grade_correct_total += 1
                bucket["grade_correct_count"] += 1

        confidence = prediction.get("confidence") if prediction else None
        if class_result is False and isinstance(confidence, (int, float)) and confidence >= 0.90:
            high_confidence_override_total += 1
            bucket["high_confidence_override_count"] += 1

        manual_review = prediction.get("manual_review_required") if prediction else None
        if manual_review:
            bucket["manual_review_feedback_count"] += 1

    daily_rows = []
    for date_key in sorted(daily):
        row = daily[date_key]
        daily_rows.append(
            {
                "date": date_key,
                **row,
                "class_accuracy_proxy": _safe_ratio(row["class_correct_count"], row["class_labelled_count"]),
                "grade_accuracy_proxy": _safe_ratio(row["grade_correct_count"], row["grade_labelled_count"]),
                "acceptance_rate": _safe_ratio(row["accepted_count"], row["feedback_count"]),
                "override_rate": _safe_ratio(row["override_count"], row["feedback_count"]),
            }
        )

    class_rows = [
        {
            "predicted_class": class_name,
            **counts,
            "class_accuracy_proxy": _safe_ratio(counts["correct_count"], counts["labelled_count"]),
        }
        for class_name, counts in sorted(by_class.items())
    ]
    model_rows = [
        {
            "model_name": model_name,
            **counts,
            "class_accuracy_proxy": _safe_ratio(counts["correct_count"], counts["labelled_count"]),
        }
        for model_name, counts in sorted(by_model.items())
    ]

    return {
        "monitoring_type": "human_feedback_accuracy_proxy",
        "limitations": [
            "feedback_is_sparse_and_may_be_biased",
            "accepted_or_overridden_feedback_is_not_a_controlled_ground_truth_test_set",
            "automatic_retraining_is_not_performed",
        ],
        "prediction_log_count": len(predictions),
        "feedback_log_count": len(feedback),
        "labelled_class_feedback_count": labelled_class_total,
        "class_accuracy_proxy": _safe_ratio(class_correct_total, labelled_class_total),
        "labelled_grade_feedback_count": labelled_grade_total,
        "grade_accuracy_proxy": _safe_ratio(grade_correct_total, labelled_grade_total),
        "accepted_feedback_count": accepted_total,
        "override_feedback_count": override_total,
        "override_rate": _safe_ratio(override_total, len(feedback)),
        "high_confidence_override_count": high_confidence_override_total,
        "daily": daily_rows,
        "by_predicted_class": class_rows,
        "by_model": model_rows,
    }


def export_feedback_monitoring_summary(
    logs_dir: Path = LOGS_DIR,
    output_dir: Path = OUTPUTS_DIR / "final_evaluation",
) -> dict[str, Any]:
    summary = build_feedback_monitoring_summary(logs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "feedback_monitoring_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    daily_path = output_dir / "feedback_accuracy_over_time.csv"
    daily_rows = summary["daily"]
    if daily_rows:
        headers = list(daily_rows[0].keys())
        lines = [",".join(headers)]
        for row in daily_rows:
            lines.append(",".join("" if row[h] is None else str(row[h]) for h in headers))
        daily_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        daily_path.write_text(
            "date,feedback_count,class_accuracy_proxy,grade_accuracy_proxy,override_rate\n",
            encoding="utf-8",
        )

    return summary


if __name__ == "__main__":
    exported = export_feedback_monitoring_summary()
    print(json.dumps(exported, indent=2))
