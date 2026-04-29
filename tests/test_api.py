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
            "confidence_score": 0.98,
            "freshness_score": 0.02,
            "quality_grade": "Reject",
            "recommended_action": "Reject from sale and route for disposal or supplier review.",
            "reason_codes": ["ROTTEN_PREDICTION", "HIGH_CONFIDENCE"],
            "manual_review_required": False,
            "top_predictions": [{"class_name": "Banana__Rotten", "confidence": 0.98}],
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
    assert body["manual_review_required"] is False

    log_records = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_records) == 1
    logged = json.loads(log_records[0])
    assert logged["prediction_id"] == body["prediction_id"]
    assert logged["predicted_class"] == "Banana__Rotten"


def test_feedback_logs_record(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "LOGS_DIR", tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/feedback",
        json={
            "prediction_id": "pred-1",
            "predicted_class": "Banana__Rotten",
            "corrected_class": "Banana__Healthy",
            "model_name": "demo_model",
            "user_note": "manual override",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "logged"}
    log_records = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_records) == 1
    assert json.loads(log_records[0])["prediction_id"] == "pred-1"
