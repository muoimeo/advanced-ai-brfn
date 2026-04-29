from src.inference.postprocess import (
    build_business_prediction,
    quality_grade_from_prediction,
    split_class_name,
)


def test_split_class_name_extracts_produce_and_status():
    assert split_class_name("Banana__Rotten") == ("Banana", "rotten")


def test_rotten_high_confidence_maps_to_reject_with_reason_codes():
    prediction = build_business_prediction("Banana__Rotten", 0.95)

    assert prediction["produce_type"] == "Banana"
    assert prediction["freshness_status"] == "rotten"
    assert prediction["quality_grade"] == "Reject"
    assert prediction["manual_review_required"] is False
    assert "ROTTEN_PREDICTION" in prediction["reason_codes"]
    assert "HIGH_CONFIDENCE_REJECTION" in prediction["reason_codes"]


def test_low_confidence_prediction_requires_manual_review():
    prediction = build_business_prediction("Apple__Healthy", 0.42)

    assert prediction["quality_grade"] == "Review"
    assert prediction["manual_review_required"] is True
    assert "LOW_CONFIDENCE" in prediction["reason_codes"]
    assert "MANUAL_REVIEW_REQUIRED" in prediction["reason_codes"]


def test_quality_grade_thresholds():
    assert quality_grade_from_prediction("healthy", 0.91) == "A"
    assert quality_grade_from_prediction("healthy", 0.75) == "B"
    assert quality_grade_from_prediction("rotten", 0.91) == "Reject"
    assert quality_grade_from_prediction("rotten", 0.75) == "Likely reject"
