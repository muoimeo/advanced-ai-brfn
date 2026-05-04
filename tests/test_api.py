import json

from fastapi.testclient import TestClient

from src.api import main


def test_health_reports_model_loaded(monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda: object())
    client = TestClient(main.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_returns_metadata(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_metadata",
        lambda: {
            "model_name": "demo_model",
            "base_model_family": "ResNet50",
            "num_classes": 28,
            "image_size": [224, 224],
            "metrics": {"test_accuracy": 0.97},
        },
    )
    client = TestClient(main.app)

    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "demo_model"
    assert body["num_classes"] == 28


def test_predict_rejects_unsupported_file_type():
    client = TestClient(main.app)

    response = client.post(
        "/predict",
        files={"file": ("sample.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_predict_returns_schema_and_logs_prediction(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(
        main,
        "predict_image_bytes",
        lambda image_bytes: {
            "model_name": "demo_model",
            "model_version": "2026-04-29T10:00:00",
            "produce_type": "Banana",
            "predicted_class": "Banana__Rotten",
            "freshness_status": "rotten",
            "confidence": 0.98,
            "top1_top2_margin": 0.91,
            "confidence_score": 0.98,
            "freshness_score": 0.02,
            "quality_grade": "C",
            "recommended_action": "manual_review_or_discard",
            "reason_codes": ["ROTTEN_PREDICTION", "high_confidence_rotten_prediction"],
            "manual_review_required": False,
            "top_predictions": [{"class_name": "Banana__Rotten", "confidence": 0.98}],
            "prediction": {
                "predicted_class": "Banana__Rotten",
                "product_type": "Banana",
                "condition": "rotten",
                "confidence": 0.98,
                "top_k": [{"class_name": "Banana__Rotten", "probability": 0.98}],
                "top1_top2_margin": 0.91,
            },
            "quality": {
                "grade": "C",
                "overall_quality_score": 12.0,
                "component_scores": {"model_condition": 2.0},
                "action": "manual_review_or_discard",
                "inventory_status": "blocked_pending_review",
                "discount_percentage": None,
                "manual_review": False,
                "reason_codes": ["high_confidence_rotten_prediction"],
                "warnings": ["quality_grade_is_rule_based_not_supervised_model_output"],
            },
            "xai": {"method": "Grad-CAM", "available": False, "heatmap_path": None},
            "model_info": {
                "model_name": "demo_model",
                "model_version": "2026-04-29T10:00:00",
            },
        },
    )
    client = TestClient(main.app)

    response = client.post(
        "/predict",
        files={"file": ("sample.png", b"fake image bytes", "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_id"]
    assert body["predicted_class"] == "Banana__Rotten"
    assert body["top1_top2_margin"] == 0.91
    assert body["quality"]["grade"] == "C"
    assert body["prediction"]["condition"] == "rotten"
    assert body["manual_review_required"] is False

    log_records = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_records) == 1
    logged = json.loads(log_records[0])
    assert logged["prediction_id"] == body["prediction_id"]
    assert logged["predicted_class"] == "Banana__Rotten"
    assert logged["top1_top2_margin"] == 0.91
    assert logged["quality"]["grade"] == "C"


def test_feedback_logs_record(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "LOGS_DIR", tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/feedback",
        json={
            "prediction_id": "pred-1",
            "predicted_class": "Banana__Rotten",
            "corrected_class": "Banana__Healthy",
            "predicted_grade": "C",
            "producer_override_grade": "Review",
            "override_reason": "visible quality needs manual inspection",
            "accepted_ai_recommendation": False,
            "quality_decision_snapshot": {
                "overall_quality_score": 12.0,
                "component_scores": {"model_condition": 2.0},
                "reason_codes": ["high_confidence_rotten_prediction"],
            },
            "model_name": "demo_model",
            "user_note": "manual override",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "logged"}
    log_records = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_records) == 1
    assert json.loads(log_records[0])["prediction_id"] == "pred-1"
    assert json.loads(log_records[0])["producer_override_grade"] == "Review"


def test_feedback_monitoring_summary_reports_accuracy_proxy(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "LOGS_DIR", tmp_path)
    (tmp_path / "predictions.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prediction_id": "pred-1",
                        "created_at": "2026-05-01T10:00:00",
                        "model_name": "demo_model",
                        "predicted_class": "Banana__Rotten",
                        "quality_grade": "C",
                        "confidence": 0.98,
                        "manual_review_required": False,
                    }
                ),
                json.dumps(
                    {
                        "prediction_id": "pred-2",
                        "created_at": "2026-05-01T11:00:00",
                        "model_name": "demo_model",
                        "predicted_class": "Apple__Healthy",
                        "quality_grade": "A",
                        "confidence": 0.91,
                        "manual_review_required": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "feedback.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prediction_id": "pred-1",
                        "created_at": "2026-05-01T12:00:00",
                        "predicted_class": "Banana__Rotten",
                        "predicted_grade": "C",
                        "producer_override_class": "Banana__Healthy",
                        "producer_override_grade": "Review",
                        "accepted_ai_recommendation": False,
                    }
                ),
                json.dumps(
                    {
                        "prediction_id": "pred-2",
                        "created_at": "2026-05-01T12:05:00",
                        "predicted_class": "Apple__Healthy",
                        "predicted_grade": "A",
                        "accepted_ai_recommendation": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(main.app)

    response = client.get("/monitoring/feedback-summary")

    assert response.status_code == 200
    body = response.json()
    assert body["monitoring_type"] == "human_feedback_accuracy_proxy"
    assert body["feedback_log_count"] == 2
    assert body["labelled_class_feedback_count"] == 2
    assert body["class_accuracy_proxy"] == 0.5
    assert body["grade_accuracy_proxy"] == 0.5
    assert body["override_rate"] == 0.5
    assert body["high_confidence_override_count"] == 1
