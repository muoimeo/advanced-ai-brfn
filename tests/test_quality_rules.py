from src.quality.rules import build_quality_decision
from src.quality.schemas import ImageQualityFeatures, ModelPrediction, TopKPrediction


def features(
    image_quality_score=90.0,
    color_score=90.0,
    defect_absence_score=90.0,
    size_proxy_score=90.0,
    dark_ratio=0.02,
):
    return ImageQualityFeatures(
        image_quality_score=image_quality_score,
        color_score=color_score,
        defect_absence_score=defect_absence_score,
        size_proxy_score=size_proxy_score,
        dark_ratio=dark_ratio,
        accepted_color_ratio=color_score / 100.0,
        blur_score=image_quality_score,
        brightness_score=image_quality_score,
        foreground_area_ratio=0.42,
        feature_warnings=["size_proxy_is_relative_to_image_area_not_physical_size"],
    )


def prediction(
    class_name="Apple__Healthy",
    product_type="Apple",
    condition="healthy",
    confidence=0.92,
    margin=0.70,
    top_k=None,
):
    return ModelPrediction(
        predicted_class=class_name,
        product_type=product_type,
        condition=condition,
        confidence=confidence,
        top_k=top_k or [TopKPrediction(class_name=class_name, probability=confidence)],
        top1_top2_margin=margin,
    )


def test_high_confidence_rotten_maps_to_grade_c():
    decision = build_quality_decision(
        prediction("Banana__Rotten", "Banana", "rotten", 0.91),
        features(),
    )

    assert decision.grade == "C"
    assert decision.action == "manual_review_or_discard"
    assert "high_confidence_rotten_prediction" in decision.reason_codes


def test_high_confidence_rotten_stays_grade_c_with_poor_image_quality():
    decision = build_quality_decision(
        prediction("Banana__Rotten", "Banana", "rotten", 0.91),
        features(image_quality_score=30.0),
    )

    assert decision.grade == "C"
    assert decision.manual_review is True
    assert "image_quality_warning" in decision.reason_codes


def test_medium_confidence_rotten_maps_to_grade_c_with_manual_review():
    decision = build_quality_decision(
        prediction("Banana__Rotten", "Banana", "rotten", 0.70),
        features(),
    )

    assert decision.grade == "C"
    assert decision.manual_review is True
    assert decision.action == "manual_review_before_listing"


def test_low_confidence_prediction_maps_to_review():
    decision = build_quality_decision(prediction(confidence=0.52), features())

    assert decision.grade == "Review"
    assert decision.manual_review is True
    assert "low_model_confidence" in decision.reason_codes


def test_low_top1_top2_margin_requires_manual_review():
    decision = build_quality_decision(prediction(margin=0.04), features())

    assert decision.grade == "A"
    assert decision.manual_review is True
    assert "low_top1_top2_margin" in decision.reason_codes


def test_opposite_same_produce_condition_conflict_requires_manual_review():
    top_k = [
        TopKPrediction("Apple__Healthy", 0.52),
        TopKPrediction("Apple__Rotten", 0.43),
    ]
    decision = build_quality_decision(
        prediction(confidence=0.52, margin=0.09, top_k=top_k),
        features(),
    )

    assert decision.manual_review is True
    assert "opposite_condition_conflict" in decision.reason_codes


def test_very_poor_image_quality_maps_to_review():
    decision = build_quality_decision(
        prediction(),
        features(image_quality_score=10.0),
    )

    assert decision.grade == "Review"
    assert "very_poor_image_quality" in decision.reason_codes


def test_image_quality_warning_does_not_force_healthy_review():
    decision = build_quality_decision(
        prediction(),
        features(image_quality_score=30.0),
    )

    assert decision.grade == "B"
    assert decision.manual_review is True
    assert "image_quality_warning" in decision.reason_codes


def test_healthy_strong_features_maps_to_grade_a():
    decision = build_quality_decision(prediction(), features())

    assert decision.grade == "A"
    assert decision.manual_review is False


def test_healthy_moderate_features_maps_to_grade_b():
    decision = build_quality_decision(
        prediction(confidence=0.88),
        features(color_score=78.0, defect_absence_score=82.0, size_proxy_score=75.0),
    )

    assert decision.grade == "B"
    assert decision.discount_percentage == 20


def test_high_dark_ratio_maps_to_grade_c():
    decision = build_quality_decision(
        prediction(),
        features(color_score=62.0, dark_ratio=0.40),
    )

    assert decision.grade == "C"
