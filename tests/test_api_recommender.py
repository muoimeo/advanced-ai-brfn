from fastapi.testclient import TestClient

from src.api import main


def test_recommend_reorder_endpoint_returns_json(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_reorder_recommendations",
        lambda customer_id, top_k, method: {
            "customer_id": customer_id,
            "recommendation_date": "2026-05-02",
            "method": method,
            "top_k": top_k,
            "recommendations": [
                {
                    "customer_id": customer_id,
                    "customer_type": "family",
                    "method": method,
                    "rank": 1,
                    "product_id": "P000001",
                    "product_name": "Tomatoes",
                    "producer_id": "PR000001",
                    "score": 0.91,
                    "reason_codes": ["frequently_ordered_by_customer", "ordered_recently"],
                    "reason_text": "Recommended because it is ordered repeatedly by this customer.",
                }
            ],
            "limitations": ["recommendations_based_on_synthetic_seed_data"],
        },
    )
    client = TestClient(main.app)

    response = client.get("/recommend/reorder?customer_id=C000001&top_k=3")

    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "C000001"
    assert body["method"] == "frequency_recency"
    assert body["recommendations"][0]["product_id"] == "P000001"
    assert body["recommendations"][0]["reason_codes"]


def test_recommend_reorder_rejects_unknown_method():
    client = TestClient(main.app)

    response = client.get(
        "/recommend/reorder?customer_id=C000001&method=unknown_method"
    )

    assert response.status_code == 400
