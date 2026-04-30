import pandas as pd

from src.inference.custom_validation import (
    build_custom_validation_summary,
    build_custom_validation_table,
)


def test_custom_validation_table_joins_manifest_and_marks_correctness():
    predictions = pd.DataFrame(
        [
            {
                "image_path": "/mnt/d/project/data/raw/custom_test_images/banana.jpg",
                "model": "EfficientNetB0 final",
                "predicted_class": "Banana__Rotten",
                "confidence": 0.88,
                "top1_top2_margin": 0.32,
            }
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "image_path": "data/raw/custom_test_images/banana.jpg",
                "expected_class": "Banana__Rotten",
                "produce_type": "Banana",
                "freshness_status": "rotten",
                "notes": "external image",
            }
        ]
    )

    validation = build_custom_validation_table(predictions, manifest)

    assert validation.loc[0, "image_id"] == "banana"
    assert validation.loc[0, "expected_class"] == "Banana__Rotten"
    assert bool(validation.loc[0, "produce_correct"]) is True
    assert bool(validation.loc[0, "freshness_correct"]) is True
    assert bool(validation.loc[0, "full_class_correct"]) is True
    assert "ROTTEN_PREDICTION" in validation.loc[0, "reason_codes"]


def test_custom_validation_summary_uses_labeled_rows_only():
    validation = pd.DataFrame(
        [
            {
                "model": "EfficientNetB0 final",
                "image_id": "a",
                "expected_class": "Apple__Healthy",
                "full_class_correct": True,
                "produce_correct": True,
                "freshness_correct": True,
                "manual_review_required": False,
                "confidence": 0.90,
            },
            {
                "model": "EfficientNetB0 final",
                "image_id": "b",
                "expected_class": None,
                "full_class_correct": None,
                "produce_correct": None,
                "freshness_correct": None,
                "manual_review_required": True,
                "confidence": 0.50,
            },
        ]
    )

    summary = build_custom_validation_summary(validation)

    assert summary.loc[0, "labeled_predictions"] == 1
    assert summary.loc[0, "full_class_accuracy"] == 1.0
    assert summary.loc[0, "mean_confidence"] == 0.90
