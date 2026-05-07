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


def test_producer_forecast_endpoint_returns_rows(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_producer_forecast",
        lambda producer_id, top_k: {
            "producer_id": producer_id,
            "forecast_method": "latest_3_week_moving_average",
            "top_k": top_k,
            "items": [
                {
                    "product_id": "P000084",
                    "product_name": "Heritage Rainbow Carrots",
                    "forecast_week_start": "2026-05-04",
                    "predicted_quantity_next_week": 29.0,
                    "trend_direction": "up",
                    "basis": "latest_3_week_moving_average",
                    "alert_text": "High demand expected for Heritage Rainbow Carrots next week based on recent order trends.",
                }
            ][:top_k],
            "limitations": ["feature_refresh_not_model_retraining"],
        },
    )
    client = TestClient(main.app)

    response = client.get("/producer/forecast?producer_id=PR000003&top_k=1")

    assert response.status_code == 200
    body = response.json()
    assert body["producer_id"] == "PR000003"
    assert body["forecast_method"] == "latest_3_week_moving_average"
    assert body["items"][0]["trend_direction"] == "up"
    assert body["items"][0]["alert_text"]


def test_producer_forecast_endpoint_rejects_invalid_top_k():
    client = TestClient(main.app)

    response = client.get("/producer/forecast?producer_id=PR000003&top_k=0")

    assert response.status_code == 422


def test_producer_forecast_endpoint_returns_404_for_unknown_producer(monkeypatch):
    def _raise_unknown(producer_id, top_k):
        from src.recommender.data_loader import Task1DataError

        raise Task1DataError(f"Unknown producer_id: {producer_id}")

    monkeypatch.setattr(main, "get_producer_forecast", _raise_unknown)
    client = TestClient(main.app)

    response = client.get("/producer/forecast?producer_id=PR999999")

    assert response.status_code == 404
    assert "Unknown producer_id" in response.json()["detail"]


def test_recommend_ingest_order_endpoint_accepts_event(monkeypatch):
    monkeypatch.setattr(
        main,
        "ingest_order_event",
        lambda record: {
            "status": "ingested",
            "order_id": record["order_id"],
            "customer_id": record["customer_id"],
            "ingested_order_lines": len(record["items"]),
            "total_orders": 484,
            "total_order_lines": 1795,
            "event_log": "outputs/logs/recommender_order_events.jsonl",
            "limitations": ["advanced_ai_service_does_not_access_desd_database"],
        },
    )
    client = TestClient(main.app)

    response = client.post(
        "/recommend/ingest-order",
        json={
            "order_id": "O-LIVE-001",
            "customer_id": "C000001",
            "order_date": "2026-05-05",
            "items": [
                {
                    "product_id": "P000001",
                    "quantity": 2,
                    "unit_price": 3.5,
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert body["order_id"] == "O-LIVE-001"
    assert body["ingested_order_lines"] == 1


def test_recommend_ingest_order_endpoint_rejects_invalid_event():
    client = TestClient(main.app)

    response = client.post(
        "/recommend/ingest-order",
        json={
            "order_id": "O-LIVE-001",
            "customer_id": "C000001",
            "order_date": "2026-05-05",
            "items": [],
        },
    )

    assert response.status_code == 422


def test_recommend_ingest_history_endpoint_accepts_batch(monkeypatch):
    monkeypatch.setattr(
        main,
        "ingest_history_event",
        lambda record: {
            "status": "ingested",
            "history_batch_id": "HIST-TEST",
            "customer_id": record["customer_id"],
            "ingested_orders": len(record["orders"]),
            "ingested_order_lines": sum(len(order["items"]) for order in record["orders"]),
            "total_orders": 486,
            "total_order_lines": 1798,
            "event_log": "outputs/logs/recommender_history_events.jsonl",
            "limitations": ["advanced_ai_service_does_not_access_desd_database"],
        },
    )
    client = TestClient(main.app)

    response = client.post(
        "/recommend/ingest-history",
        json={
            "customer_id": "C000001",
            "customer_type": "family",
            "postcode_area": "BS1",
            "source": "desd_history_backfill",
            "orders": [
                {
                    "order_id": "O-HIST-001",
                    "order_date": "2026-05-05",
                    "items": [
                        {
                            "product_id": "P000001",
                            "quantity": 2,
                            "unit_price": 3.5,
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert body["customer_id"] == "C000001"
    assert body["ingested_orders"] == 1
    assert body["ingested_order_lines"] == 1


def test_recommend_ingest_history_endpoint_rejects_empty_orders():
    client = TestClient(main.app)

    response = client.post(
        "/recommend/ingest-history",
        json={
            "customer_id": "C000001",
            "orders": [],
        },
    )

    assert response.status_code == 422


def test_catalog_ingest_producer_endpoint_accepts_event(monkeypatch):
    monkeypatch.setattr(
        main,
        "ingest_producer_event",
        lambda record: {
            "status": "ingested",
            "producer_id": record["producer_id"],
            "catalog_producers": 13,
            "event_log": "outputs/logs/catalog_producer_events.jsonl",
            "limitations": ["catalogue_events_update_metadata_without_model_retraining"],
        },
    )
    client = TestClient(main.app)

    response = client.post(
        "/catalog/ingest-producer",
        json={
            "event_type": "producer_upserted",
            "producer_id": "PR000020",
            "producer_name": "New Bristol Bakery",
            "postcode_area": "BS1",
            "categories": ["bakery"],
            "organic_certified": False,
            "created_at": "2026-05-06T10:30:00Z",
            "source": "desd_producer_event",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert body["producer_id"] == "PR000020"


def test_catalog_ingest_producer_endpoint_rejects_invalid_event_type():
    client = TestClient(main.app)

    response = client.post(
        "/catalog/ingest-producer",
        json={
            "event_type": "producer_deleted",
            "producer_id": "PR000020",
            "producer_name": "New Bristol Bakery",
            "postcode_area": "BS1",
            "created_at": "2026-05-06T10:30:00Z",
            "source": "desd_producer_event",
        },
    )

    assert response.status_code == 422


def test_catalog_ingest_product_endpoint_accepts_event(monkeypatch):
    monkeypatch.setattr(
        main,
        "ingest_product_event",
        lambda record: {
            "status": "ingested",
            "product_id": record["product_id"],
            "producer_id": record["producer_id"],
            "available": record["available"],
            "catalog_products": 61,
            "event_log": "outputs/logs/catalog_product_events.jsonl",
            "limitations": ["catalogue_events_update_metadata_without_model_retraining"],
        },
    )
    client = TestClient(main.app)

    response = client.post(
        "/catalog/ingest-product",
        json={
            "event_type": "product_upserted",
            "product_id": "P000001",
            "producer_id": "PR000001",
            "product_name": "Organic Tomatoes",
            "category": "vegetables",
            "unit": "kg",
            "price": 3.5,
            "seasonal": True,
            "seasonal_start_month": 5,
            "seasonal_end_month": 10,
            "available": True,
            "created_at": "2026-05-06T10:30:00Z",
            "source": "desd_product_event",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert body["product_id"] == "P000001"


def test_catalog_ingest_product_endpoint_rejects_missing_required_field():
    client = TestClient(main.app)

    response = client.post(
        "/catalog/ingest-product",
        json={
            "event_type": "product_upserted",
            "product_id": "P000001",
            "producer_id": "PR000001",
        },
    )

    assert response.status_code == 422


def test_catalog_ingest_product_endpoint_rejects_invalid_event_type():
    client = TestClient(main.app)

    response = client.post(
        "/catalog/ingest-product",
        json={
            "event_type": "product_deleted",
            "product_id": "P000001",
            "producer_id": "PR000001",
            "product_name": "Organic Tomatoes",
            "category": "vegetables",
            "unit": "kg",
            "price": 3.5,
            "seasonal": True,
            "seasonal_start_month": 5,
            "seasonal_end_month": 10,
            "available": True,
            "created_at": "2026-05-06T10:30:00Z",
            "source": "desd_product_event",
        },
    )

    assert response.status_code == 422
