from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import OUTPUTS_DIR


FINAL_EXPERIMENT_SLUG = "efficientnetb0_aug_oversampled_finetuned_wsl"
DEFAULT_PREDICTIONS_PATH = (
    OUTPUTS_DIR / "final_evaluation" / "all_cv_model_confidence_audit_predictions.csv"
)
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "final_evaluation"

CONFIDENCE_REVIEW_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MARGIN_REVIEW_THRESHOLDS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
ROTTEN_AUTO_BLOCK_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]

COST_MODEL = {
    "correct_auto_decision": 0,
    "manual_review": 1,
    "wrong_produce_same_condition": 2,
    "healthy_predicted_rotten": 3,
    "rotten_predicted_healthy": 10,
}


def _condition(class_name: str) -> str:
    text = str(class_name).lower()
    if text.endswith("__healthy"):
        return "healthy"
    if text.endswith("__rotten"):
        return "rotten"
    return "unknown"


def _produce(class_name: str) -> str:
    return str(class_name).split("__", maxsplit=1)[0]


def prediction_cost(true_class: str, predicted_class: str) -> tuple[int, str]:
    true_condition = _condition(true_class)
    predicted_condition = _condition(predicted_class)

    if true_class == predicted_class:
        return COST_MODEL["correct_auto_decision"], "correct_auto_decision"
    if true_condition == "rotten" and predicted_condition == "healthy":
        return COST_MODEL["rotten_predicted_healthy"], "rotten_predicted_healthy"
    if true_condition == "healthy" and predicted_condition == "rotten":
        return COST_MODEL["healthy_predicted_rotten"], "healthy_predicted_rotten"
    if true_condition == predicted_condition or _produce(true_class) != _produce(predicted_class):
        return COST_MODEL["wrong_produce_same_condition"], "wrong_produce_same_condition"
    return COST_MODEL["wrong_produce_same_condition"], "wrong_produce_same_condition"


def _manual_review_required(
    row: pd.Series,
    confidence_review_threshold: float,
    margin_review_threshold: float,
    rotten_auto_block_threshold: float,
) -> bool:
    confidence = float(row["confidence"])
    if confidence < confidence_review_threshold:
        return True
    if "top1_top2_margin" in row and pd.notna(row.get("top1_top2_margin")):
        if float(row["top1_top2_margin"]) < margin_review_threshold:
            return True
    if _condition(str(row["predicted_class"])) == "rotten" and confidence < rotten_auto_block_threshold:
        return True
    return False


def evaluate_threshold_setting(
    predictions: pd.DataFrame,
    confidence_review_threshold: float,
    margin_review_threshold: float,
    rotten_auto_block_threshold: float,
) -> dict[str, Any]:
    total_cost = 0
    manual_review_count = 0
    false_fresh_count = 0
    false_rotten_count = 0
    wrong_produce_count = 0
    correct_auto_count = 0

    for _, row in predictions.iterrows():
        if _manual_review_required(
            row,
            confidence_review_threshold,
            margin_review_threshold,
            rotten_auto_block_threshold,
        ):
            total_cost += COST_MODEL["manual_review"]
            manual_review_count += 1
            continue

        cost, reason = prediction_cost(str(row["class_name"]), str(row["predicted_class"]))
        total_cost += cost
        if reason == "correct_auto_decision":
            correct_auto_count += 1
        elif reason == "rotten_predicted_healthy":
            false_fresh_count += 1
        elif reason == "healthy_predicted_rotten":
            false_rotten_count += 1
        else:
            wrong_produce_count += 1

    total = len(predictions)
    auto_count = total - manual_review_count
    return {
        "confidence_review_threshold": confidence_review_threshold,
        "margin_review_threshold": margin_review_threshold,
        "rotten_auto_block_threshold": rotten_auto_block_threshold,
        "expected_operational_cost": round(total_cost / total, 6) if total else 0.0,
        "total_operational_cost": total_cost,
        "false_fresh_count": false_fresh_count,
        "false_rotten_count": false_rotten_count,
        "wrong_produce_count": wrong_produce_count,
        "correct_auto_count": correct_auto_count,
        "manual_review_count": manual_review_count,
        "manual_review_rate": round(manual_review_count / total, 6) if total else 0.0,
        "auto_decision_rate": round(auto_count / total, 6) if total else 0.0,
        "evaluated_rows": total,
        "note": "Cost-sensitive deployment proxy; not a retrained model.",
    }


def run_threshold_grid(
    predictions: pd.DataFrame,
    confidence_thresholds: list[float] | None = None,
    margin_thresholds: list[float] | None = None,
    rotten_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    rows = []
    for confidence_threshold, margin_threshold, rotten_threshold in product(
        confidence_thresholds or CONFIDENCE_REVIEW_THRESHOLDS,
        margin_thresholds or MARGIN_REVIEW_THRESHOLDS,
        rotten_thresholds or ROTTEN_AUTO_BLOCK_THRESHOLDS,
    ):
        rows.append(
            evaluate_threshold_setting(
                predictions,
                confidence_threshold,
                margin_threshold,
                rotten_threshold,
            )
        )
    return pd.DataFrame(rows)


def select_operating_point(results: pd.DataFrame) -> dict[str, Any]:
    if results.empty:
        return {}
    sorted_results = results.sort_values(
        [
            "expected_operational_cost",
            "false_fresh_count",
            "manual_review_rate",
        ],
        ascending=[True, True, True],
    )
    selected = sorted_results.iloc[0].to_dict()
    selected["selection_rule"] = (
        "minimum expected cost; tie-break lower false-fresh count; "
        "tie-break lower manual-review rate"
    )
    return selected


def export_cost_sensitive_threshold_outputs(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    experiment_slug: str = FINAL_EXPERIMENT_SLUG,
) -> dict[str, str]:
    predictions = pd.read_csv(predictions_path)
    predictions = predictions[predictions["experiment_slug"] == experiment_slug].copy()
    results = run_threshold_grid(predictions)
    selected = select_operating_point(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "cost_sensitive_threshold_results.csv"
    summary_path = output_dir / "cost_sensitive_threshold_summary.json"
    results.to_csv(results_path, index=False)
    summary = {
        "experiment_slug": experiment_slug,
        "cost_model": COST_MODEL,
        "selected_operating_point": selected,
        "evaluated_settings": int(len(results)),
        "input_predictions": str(predictions_path),
        "limitations": [
            "threshold study uses existing test predictions and does not retrain the model",
            "margin threshold is only active when top1_top2_margin is available",
            "cost values are explicit coursework assumptions for deployment-risk analysis",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "cost_sensitive_threshold_results.csv": str(results_path),
        "cost_sensitive_threshold_summary.json": str(summary_path),
    }


if __name__ == "__main__":
    written = export_cost_sensitive_threshold_outputs()
    for filename, path in written.items():
        print(f"{filename}: {path}")
