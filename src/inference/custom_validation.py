from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.inference.postprocess import build_business_prediction, split_class_name


def image_id_from_path(path: str) -> str:
    normalized = str(path).replace("\\", "/")
    return Path(normalized).stem


def _filename_from_path(path: str) -> str:
    normalized = str(path).replace("\\", "/")
    return Path(normalized).name


def _correct_or_missing(actual: str, expected: str | None) -> bool | None:
    if expected is None or pd.isna(expected) or expected == "":
        return None
    return actual == expected


def _split_optional_class_name(class_name: str | None) -> tuple[str, str]:
    if class_name is None or pd.isna(class_name) or class_name == "":
        return "", ""
    return split_class_name(class_name)


def build_custom_validation_table(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    predictions = predictions.copy()
    manifest = manifest.copy()

    predictions["image_filename"] = predictions["image_path"].map(_filename_from_path)
    predictions["image_id"] = predictions["image_path"].map(image_id_from_path)
    manifest["image_filename"] = manifest["image_path"].map(_filename_from_path)

    validation = predictions.merge(
        manifest[
            [
                "image_filename",
                "expected_class",
                "produce_type",
                "freshness_status",
                "notes",
            ]
        ],
        on="image_filename",
        how="left",
        suffixes=("", "_expected"),
    )

    expected_parts = validation["expected_class"].map(_split_optional_class_name)
    predicted_parts = validation["predicted_class"].fillna("").map(split_class_name)

    validation["expected_produce_type"] = [item[0] for item in expected_parts]
    validation["expected_freshness_status"] = [item[1] for item in expected_parts]
    validation["predicted_produce_type"] = [item[0] for item in predicted_parts]
    validation["predicted_freshness_status"] = [item[1] for item in predicted_parts]

    validation["produce_correct"] = [
        _correct_or_missing(actual, expected)
        for actual, expected in zip(
            validation["predicted_produce_type"],
            validation["expected_produce_type"],
        )
    ]
    validation["freshness_correct"] = [
        _correct_or_missing(actual, expected)
        for actual, expected in zip(
            validation["predicted_freshness_status"],
            validation["expected_freshness_status"],
        )
    ]
    validation["full_class_correct"] = [
        _correct_or_missing(actual, expected)
        for actual, expected in zip(
            validation["predicted_class"],
            validation["expected_class"],
        )
    ]

    if "top1_top2_margin" not in validation.columns:
        validation["top1_top2_margin"] = None

    if "manual_review_required" not in validation.columns or "reason_codes" not in validation.columns:
        business_predictions = [
            build_business_prediction(
                predicted_class=row["predicted_class"],
                confidence=float(row["confidence"]),
                top1_top2_margin=(
                    None
                    if pd.isna(row["top1_top2_margin"])
                    else float(row["top1_top2_margin"])
                ),
            )
            for _, row in validation.iterrows()
        ]
        validation["manual_review_required"] = [
            item["manual_review_required"] for item in business_predictions
        ]
        validation["reason_codes"] = [
            "|".join(item["reason_codes"]) for item in business_predictions
        ]

    optional_output_columns = [
        "quality_grade",
        "overall_quality_score",
        "component_scores",
        "recommended_action",
        "inventory_status",
        "discount_percentage",
        "warnings",
    ]

    output_columns = [
        "image_id",
        "image_path",
        "expected_class",
        "expected_produce_type",
        "expected_freshness_status",
        "model",
        "predicted_class",
        "predicted_produce_type",
        "predicted_freshness_status",
        "confidence",
        "top1_top2_margin",
        "produce_correct",
        "freshness_correct",
        "full_class_correct",
        "manual_review_required",
        "reason_codes",
        *[
            column
            for column in optional_output_columns
            if column in validation.columns
        ],
        "notes",
    ]
    return validation[output_columns]


def build_custom_validation_summary(validation: pd.DataFrame) -> pd.DataFrame:
    labeled = validation.dropna(subset=["expected_class"]).copy()
    if labeled.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "labeled_predictions",
                "full_class_accuracy",
                "produce_accuracy",
                "freshness_accuracy",
                "manual_review_rate",
                "mean_confidence",
            ]
        )

    return (
        labeled.groupby("model", dropna=False)
        .agg(
            labeled_predictions=("image_id", "count"),
            full_class_accuracy=("full_class_correct", "mean"),
            produce_accuracy=("produce_correct", "mean"),
            freshness_accuracy=("freshness_correct", "mean"),
            manual_review_rate=("manual_review_required", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )


def write_custom_validation_outputs(
    predictions_csv: Path,
    manifest_csv: Path,
    validation_csv: Path,
    summary_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(predictions_csv)
    manifest = pd.read_csv(manifest_csv)

    validation = build_custom_validation_table(predictions, manifest)
    summary = build_custom_validation_summary(validation)

    validation_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(validation_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    return validation, summary
