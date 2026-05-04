import pandas as pd

from src.evaluation.cost_sensitive_thresholds import (
    evaluate_threshold_setting,
    prediction_cost,
    run_threshold_grid,
    select_operating_point,
)


def test_cost_function_penalises_false_fresh_more_than_false_rotten():
    false_fresh_cost, false_fresh_reason = prediction_cost(
        "Banana__Rotten",
        "Banana__Healthy",
    )
    false_rotten_cost, false_rotten_reason = prediction_cost(
        "Banana__Healthy",
        "Banana__Rotten",
    )

    assert false_fresh_reason == "rotten_predicted_healthy"
    assert false_rotten_reason == "healthy_predicted_rotten"
    assert false_fresh_cost > false_rotten_cost


def test_manual_review_has_lower_cost_than_false_fresh():
    predictions = pd.DataFrame(
        [
            {
                "class_name": "Banana__Rotten",
                "predicted_class": "Banana__Healthy",
                "confidence": 0.40,
            }
        ]
    )

    reviewed = evaluate_threshold_setting(predictions, 0.60, 0.00, 0.80)
    auto_wrong = evaluate_threshold_setting(predictions, 0.30, 0.00, 0.80)

    assert reviewed["manual_review_count"] == 1
    assert reviewed["expected_operational_cost"] < auto_wrong["expected_operational_cost"]


def test_threshold_grid_returns_non_empty_results():
    predictions = pd.DataFrame(
        [
            {"class_name": "Apple__Healthy", "predicted_class": "Apple__Healthy", "confidence": 0.95},
            {"class_name": "Banana__Rotten", "predicted_class": "Banana__Healthy", "confidence": 0.91},
        ]
    )

    results = run_threshold_grid(
        predictions,
        confidence_thresholds=[0.50, 0.60],
        margin_thresholds=[0.00],
        rotten_thresholds=[0.80],
    )

    assert len(results) == 2
    assert {"expected_operational_cost", "false_fresh_count"} <= set(results.columns)


def test_selected_threshold_uses_cost_then_false_fresh_then_review_rate():
    results = pd.DataFrame(
        [
            {
                "expected_operational_cost": 1.0,
                "false_fresh_count": 2,
                "manual_review_rate": 0.1,
                "confidence_review_threshold": 0.5,
            },
            {
                "expected_operational_cost": 1.0,
                "false_fresh_count": 1,
                "manual_review_rate": 0.2,
                "confidence_review_threshold": 0.6,
            },
            {
                "expected_operational_cost": 1.0,
                "false_fresh_count": 1,
                "manual_review_rate": 0.1,
                "confidence_review_threshold": 0.7,
            },
        ]
    )

    selected = select_operating_point(results)

    assert selected["confidence_review_threshold"] == 0.7
