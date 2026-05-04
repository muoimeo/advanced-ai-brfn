from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.config import OUTPUTS_DIR
from src.inference.postprocess import build_business_prediction
from src.quality.image_features import extract_image_features
from src.quality.rules import (
    DEFAULT_WEIGHTS,
    WEIGHT_SETS,
    build_quality_decision,
    quality_decision_to_dict,
    top_k_from_dicts,
)
from src.quality.schemas import ModelPrediction


def runtime_path_to_local_path(path: str | Path) -> Path:
    text = str(path)
    candidate_paths = [Path(text)]

    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive = text[5]
        candidate_paths.append(Path(f"{drive}:{text[6:]}"))
    elif re.match(r"^[A-Za-z]:[\\/]", text):
        drive = text[0].lower()
        relative_path = text[2:].replace("\\", "/").lstrip("/")
        candidate_paths.append(Path(f"/mnt/{drive}/{relative_path}"))

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    return candidate_paths[-1]


def build_model_prediction_from_row(row: pd.Series) -> ModelPrediction:
    top_predictions = json.loads(row["top_predictions"])
    business_prediction = build_business_prediction(
        predicted_class=row["predicted_class"],
        confidence=float(row["confidence"]),
        top_predictions=top_predictions,
        top1_top2_margin=(
            None
            if pd.isna(row.get("top1_top2_margin"))
            else float(row["top1_top2_margin"])
        ),
    )
    return ModelPrediction(
        predicted_class=row["predicted_class"],
        product_type=business_prediction["produce_type"],
        condition=business_prediction["freshness_status"],
        confidence=float(row["confidence"]),
        top_k=top_k_from_dicts(top_predictions),
        top1_top2_margin=business_prediction["top1_top2_margin"],
    )


QUALITY_ABLATION_WEIGHT_SETS = {
    "full_default": DEFAULT_WEIGHTS,
    "model_only": {
        "model_condition": 1.00,
        "color": 0.00,
        "defect_absence": 0.00,
        "image_quality": 0.00,
        "size_proxy": 0.00,
    },
    "visual_only": {
        "model_condition": 0.00,
        "color": 0.40,
        "defect_absence": 0.30,
        "image_quality": 0.20,
        "size_proxy": 0.10,
    },
    "no_color": {
        "model_condition": 0.55,
        "color": 0.00,
        "defect_absence": 0.25,
        "image_quality": 0.15,
        "size_proxy": 0.05,
    },
    "no_defect_absence": {
        "model_condition": 0.55,
        "color": 0.30,
        "defect_absence": 0.00,
        "image_quality": 0.10,
        "size_proxy": 0.05,
    },
    "no_image_quality": {
        "model_condition": 0.50,
        "color": 0.30,
        "defect_absence": 0.15,
        "image_quality": 0.00,
        "size_proxy": 0.05,
    },
    "no_size_proxy": {
        "model_condition": 0.45,
        "color": 0.25,
        "defect_absence": 0.15,
        "image_quality": 0.15,
        "size_proxy": 0.00,
    },
    "more_visual_colour": WEIGHT_SETS["more_visual_colour"],
    "safer_model_led": WEIGHT_SETS["safer_model_led"],
}


def evaluate_weight_sets(
    predictions_df: pd.DataFrame,
    weight_sets: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    selected_weight_sets = weight_sets or WEIGHT_SETS
    rows = []

    for weight_set_name, weights in selected_weight_sets.items():
        grade_counts = {"A": 0, "B": 0, "C": 0, "Review": 0}
        manual_review_count = 0
        rotten_count = 0
        rotten_to_c_or_review_count = 0
        risky_a_b_decisions = 0
        overall_quality_scores = []
        action_counts: dict[str, int] = {}

        for _, row in predictions_df.iterrows():
            model_prediction = build_model_prediction_from_row(row)
            image_path = runtime_path_to_local_path(row["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(
                    "Custom validation image does not exist. "
                    f"original={row['image_path']!r}, resolved={str(image_path)!r}"
                )
            image_features = extract_image_features(
                image_bytes=image_path.read_bytes(),
                product_type=model_prediction.product_type,
            )
            decision = build_quality_decision(
                model_prediction,
                image_features,
                weights=weights,
            )
            decision_dict = quality_decision_to_dict(decision)
            grade = decision_dict["grade"]
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            overall_quality_scores.append(float(decision_dict["overall_quality_score"]))
            action = str(decision_dict["action"])
            action_counts[action] = action_counts.get(action, 0) + 1
            if decision_dict["manual_review"]:
                manual_review_count += 1

            uncertain = (
                model_prediction.confidence < 0.60
                or (
                    model_prediction.top1_top2_margin is not None
                    and model_prediction.top1_top2_margin < 0.15
                )
            )
            if model_prediction.condition == "rotten":
                rotten_count += 1
                if grade in {"C", "Review"}:
                    rotten_to_c_or_review_count += 1
                if grade in {"A", "B"}:
                    risky_a_b_decisions += 1
            elif uncertain and grade in {"A", "B"}:
                risky_a_b_decisions += 1

        total = len(predictions_df)
        rows.append(
            {
                "weight_set": weight_set_name,
                "grade_a_count": grade_counts.get("A", 0),
                "grade_b_count": grade_counts.get("B", 0),
                "grade_c_count": grade_counts.get("C", 0),
                "review_count": grade_counts.get("Review", 0),
                "manual_review_rate": (
                    manual_review_count / total if total else 0.0
                ),
                "rotten_to_c_or_review_rate": (
                    rotten_to_c_or_review_count / rotten_count
                    if rotten_count
                    else 0.0
                ),
                "risky_a_b_decisions": risky_a_b_decisions,
                "mean_overall_quality_score": (
                    sum(overall_quality_scores) / total if total else 0.0
                ),
                "action_distribution": "|".join(
                    f"{action}:{count}" for action, count in sorted(action_counts.items())
                ),
                "note": "No supervised grade labels; compare consistency and risk controls only.",
            }
        )

    return pd.DataFrame(rows)


def evaluate_quality_score_ablation(predictions_df: pd.DataFrame) -> pd.DataFrame:
    return evaluate_weight_sets(predictions_df, weight_sets=QUALITY_ABLATION_WEIGHT_SETS)


def export_quality_score_ablation(
    predictions_csv: Path = OUTPUTS_DIR
    / "xai_examples"
    / "custom_image_tests"
    / "custom_image_predictions.csv",
    quality_output_dir: Path = OUTPUTS_DIR / "quality_rule_eval",
    final_output_dir: Path = OUTPUTS_DIR / "final_evaluation",
) -> dict[str, str]:
    predictions = pd.read_csv(predictions_csv)
    result = evaluate_quality_score_ablation(predictions)
    quality_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    quality_path = quality_output_dir / "quality_score_ablation.csv"
    final_path = final_output_dir / "quality_score_ablation.csv"
    result.to_csv(quality_path, index=False)
    result.to_csv(final_path, index=False)
    return {
        "quality_score_ablation.csv": str(quality_path),
        "final_evaluation/quality_score_ablation.csv": str(final_path),
    }


if __name__ == "__main__":
    written = export_quality_score_ablation()
    for filename, path in written.items():
        print(f"{filename}: {path}")
