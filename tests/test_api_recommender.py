from fastapi.testclient import TestClient

from src.api import main


def test_recommend_reorder_endpoint_returns_json(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_reorder_recommendations",
        lambda customer_id, top_k, method, include_discovery=False: {
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
    assert "quick_reorder" not in body
    assert "you_may_also_like" not in body


def test_recommend_reorder_endpoint_can_include_discovery(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_reorder_recommendations",
        lambda customer_id, top_k, method, include_discovery=False: {
            "customer_id": customer_id,
            "recommendation_date": "2026-05-02",
            "method": "hybrid_reorder_discovery",
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
                    "reason_codes": ["frequently_ordered_by_customer"],
                    "reason_text": "Recommended because it is ordered repeatedly by this customer.",
                }
            ],
            "quick_reorder": [
                {
                    "customer_id": customer_id,
                    "customer_type": "family",
                    "method": method,
                    "rank": 1,
                    "product_id": "P000001",
                    "product_name": "Tomatoes",
                    "producer_id": "PR000001",
                    "score": 0.91,
                    "reason_codes": ["frequently_ordered_by_customer"],
                    "reason_text": "Recommended because it is ordered repeatedly by this customer.",
                }
            ],
            "you_may_also_like": [
                {
                    "customer_id": customer_id,
                    "customer_type": "family",
                    "method": "co_occurrence_discovery",
                    "rank": 1,
                    "product_id": "P000002",
                    "product_name": "Jam",
                    "producer_id": "PR000002",
                    "score": 0.77,
                    "score_components": {"cooccurrence": 0.72},
                    "based_on_product_ids": ["P000001"],
                    "reason_codes": ["commonly_bought_together", "new_to_customer"],
                    "reason_text": "Recommended because it is commonly bought with products this customer purchased.",
                }
            ],
            "limitations": ["recommendations_based_on_synthetic_seed_data"],
        },
    )
    client = TestClient(main.app)

    response = client.get(
        "/recommend/reorder?customer_id=C000001&top_k=3&include_discovery=true"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "hybrid_reorder_discovery"
    assert body["quick_reorder"][0]["product_id"] == "P000001"
    assert body["you_may_also_like"][0]["product_id"] == "P000002"
    assert body["you_may_also_like"][0]["score_components"]


def test_recommend_reorder_rejects_unknown_method():
    client = TestClient(main.app)

    response = client.get(
        "/recommend/reorder?customer_id=C000001&method=unknown_method"
    )

    assert response.status_code == 400
