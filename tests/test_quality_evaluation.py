import pandas as pd
import pytest

from src.quality import evaluation
from src.quality.schemas import ImageQualityFeatures


def test_runtime_path_to_local_path_keeps_existing_path(tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    assert evaluation.runtime_path_to_local_path(str(image_path)) == image_path


def test_runtime_path_to_local_path_converts_wsl_path_on_windows(monkeypatch):
    original_exists = evaluation.Path.exists

    def fake_exists(path):
        if str(path) == "d:\\UWE\\sample.jpg":
            return True
        return original_exists(path)

    monkeypatch.setattr(evaluation.Path, "exists", fake_exists)

    resolved = evaluation.runtime_path_to_local_path("/mnt/d/UWE/sample.jpg")

    assert str(resolved) == "d:\\UWE\\sample.jpg"


def test_runtime_path_to_local_path_converts_windows_path_on_wsl(monkeypatch):
    original_exists = evaluation.Path.exists

    def fake_exists(path):
        if str(path).replace("\\", "/") == "/mnt/d/UWE/sample.jpg":
            return True
        return original_exists(path)

    monkeypatch.setattr(evaluation.Path, "exists", fake_exists)

    resolved = evaluation.runtime_path_to_local_path(r"D:\UWE\sample.jpg")

    assert str(resolved).replace("\\", "/") == "/mnt/d/UWE/sample.jpg"


def test_evaluate_weight_sets_returns_rows_for_each_weight_set(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    predictions = pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "predicted_class": "Banana__Rotten",
                "confidence": 0.95,
                "top1_top2_margin": 0.90,
                "top_predictions": (
                    '[{"class_name": "Banana__Rotten", "confidence": 0.95}, '
                    '{"class_name": "Banana__Healthy", "confidence": 0.05}]'
                ),
            }
        ]
    )
    monkeypatch.setattr(
        evaluation,
        "extract_image_features",
        lambda image_bytes, product_type: ImageQualityFeatures(
            image_quality_score=90.0,
            color_score=80.0,
            defect_absence_score=80.0,
            size_proxy_score=90.0,
            dark_ratio=0.05,
            accepted_color_ratio=0.80,
            blur_score=90.0,
            brightness_score=90.0,
            foreground_area_ratio=0.42,
            feature_warnings=[],
        ),
    )

    result = evaluation.evaluate_weight_sets(predictions)

    assert set(result["weight_set"]) == {
        "default_balanced",
        "more_visual_colour",
        "safer_model_led",
    }
    assert result["rotten_to_c_or_review_rate"].eq(1.0).all()
    assert result["risky_a_b_decisions"].eq(0).all()


def test_evaluate_weight_sets_raises_clear_error_for_missing_image():
    predictions = pd.DataFrame(
        [
            {
                "image_path": "/missing/image.jpg",
                "predicted_class": "Banana__Rotten",
                "confidence": 0.95,
                "top1_top2_margin": 0.90,
                "top_predictions": (
                    '[{"class_name": "Banana__Rotten", "confidence": 0.95}, '
                    '{"class_name": "Banana__Healthy", "confidence": 0.05}]'
                ),
            }
        ]
    )

    with pytest.raises(FileNotFoundError, match="original='/missing/image.jpg'"):
        evaluation.evaluate_weight_sets(predictions)


def test_quality_ablation_outputs_required_profiles(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    predictions = pd.DataFrame(
        [
            {
                "image_path": str(image_path),
                "predicted_class": "Apple__Healthy",
                "confidence": 0.95,
                "top1_top2_margin": 0.90,
                "top_predictions": (
                    '[{"class_name": "Apple__Healthy", "confidence": 0.95}, '
                    '{"class_name": "Apple__Rotten", "confidence": 0.05}]'
                ),
            }
        ]
    )
    monkeypatch.setattr(
        evaluation,
        "extract_image_features",
        lambda image_bytes, product_type: ImageQualityFeatures(
            image_quality_score=90.0,
            color_score=90.0,
            defect_absence_score=90.0,
            size_proxy_score=90.0,
            dark_ratio=0.01,
            accepted_color_ratio=0.90,
            blur_score=90.0,
            brightness_score=90.0,
            foreground_area_ratio=0.42,
            feature_warnings=[],
        ),
    )

    result = evaluation.evaluate_quality_score_ablation(predictions)

    assert set(result["weight_set"]) == {
        "full_default",
        "model_only",
        "visual_only",
        "no_color",
        "no_defect_absence",
        "no_image_quality",
        "no_size_proxy",
        "more_visual_colour",
        "safer_model_led",
    }
    assert "mean_overall_quality_score" in result.columns
    assert "action_distribution" in result.columns
    assert result["note"].str.contains("No supervised grade labels").all()
