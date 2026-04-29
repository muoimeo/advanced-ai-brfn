import numpy as np

from src.inference import predictor


class FakeModel:
    def predict(self, image_tensor, verbose=0):
        probabilities = np.zeros((1, 3), dtype=np.float32)
        probabilities[0, 1] = 0.92
        probabilities[0, 0] = 0.05
        probabilities[0, 2] = 0.03
        return probabilities


def test_predict_image_bytes_builds_business_response(monkeypatch):
    monkeypatch.setattr(predictor, "get_model", lambda: FakeModel())
    monkeypatch.setattr(
        predictor,
        "get_class_names",
        lambda: ["Apple__Healthy", "Banana__Rotten", "Carrot__Healthy"],
    )
    monkeypatch.setattr(
        predictor,
        "get_metadata",
        lambda: {
            "model_name": "fake_model",
            "trained_at": "2026-04-29T10:00:00",
        },
    )
    monkeypatch.setattr(predictor, "image_bytes_to_tensor", lambda image_bytes: object())

    result = predictor.predict_image_bytes(b"fake-image-bytes", top_k=2)

    assert result["model_name"] == "fake_model"
    assert result["model_version"] == "2026-04-29T10:00:00"
    assert result["predicted_class"] == "Banana__Rotten"
    assert result["confidence"] == float(np.float32(0.92))
    assert result["quality_grade"] == "Reject"
    assert len(result["top_predictions"]) == 2
    assert "ROTTEN_PREDICTION" in result["reason_codes"]
