from __future__ import annotations

import pandas as pd
import pytest

from src.recommender.catalog_updates import append_producer_event, append_product_event
from src.recommender.catalog_updates import apply_catalog_events
from src.recommender.data_loader import Task1DataError
from src.recommender.live_updates import append_order_event
from src.recommender.pipeline import get_producer_forecast


@pytest.fixture()
def task1_data_dir(tmp_path, monkeypatch):
    pd.DataFrame(
        [
            {"customer_id": "C000001", "customer_type": "family", "postcode_area": "BS1"},
        ]
    ).to_csv(tmp_path / "customers.csv", index=False)
    pd.DataFrame(
        [
            {"producer_id": "PR000001", "producer_name": "North Farm", "postcode_area": "BS1"},
        ]
    ).to_csv(tmp_path / "producers.csv", index=False)
    pd.DataFrame(
        [
            {
                "product_id": "P000001",
                "product_name": "Tomatoes",
                "category": "vegetables",
                "producer_id": "PR000001",
                "seasonal_start_month": 1,
                "seasonal_end_month": 12,
                "base_price": 2.5,
                "unit": "kg",
                "availability_status": "available",
            },
        ]
    ).to_csv(tmp_path / "products.csv", index=False)
    pd.DataFrame(
        [
            {
                "order_id": "O000001",
                "customer_id": "C000001",
                "order_date": "2026-04-01",
                "total_amount": 5.0,
            },
        ]
    ).to_csv(tmp_path / "orders.csv", index=False)
    pd.DataFrame(
        [
            {
                "order_id": "O000001",
                "product_id": "P000001",
                "quantity": 2,
                "unit_price": 2.5,
            },
        ]
    ).to_csv(tmp_path / "order_items.csv", index=False)

    producer_log = tmp_path / "catalog_producer_events.jsonl"
    product_log = tmp_path / "catalog_product_events.jsonl"
    order_log = tmp_path / "recommender_order_events.jsonl"
    history_log = tmp_path / "recommender_history_events.jsonl"

    monkeypatch.setattr("src.recommender.catalog_updates.PRODUCER_EVENT_LOG", producer_log)
    monkeypatch.setattr("src.recommender.catalog_updates.PRODUCT_EVENT_LOG", product_log)
    monkeypatch.setattr("src.recommender.live_updates.ORDER_EVENT_LOG", order_log)
    monkeypatch.setattr("src.recommender.live_updates.HISTORY_EVENT_LOG", history_log)

    return tmp_path


def test_producer_forecast_uses_catalog_and_order_event_overlays(task1_data_dir):
    from src.recommender.data_loader import load_task1_dataset

    dataset = load_task1_dataset(task1_data_dir)
    append_producer_event(
        dataset,
        {
            "event_type": "producer_upserted",
            "producer_id": "PR000020",
            "producer_name": "New Bristol Bakery",
            "postcode_area": "BS1",
            "categories": ["bakery"],
            "organic_certified": False,
            "created_at": "2026-05-06T10:00:00Z",
            "source": "desd_producer_event",
        },
    )
    append_product_event(
        dataset,
        {
            "event_type": "product_upserted",
            "product_id": "P000020",
            "producer_id": "PR000020",
            "product_name": "Walnut Bread",
            "category": "bakery",
            "unit": "loaf",
            "price": 4.2,
            "seasonal": False,
            "seasonal_start_month": None,
            "seasonal_end_month": None,
            "available": True,
            "created_at": "2026-05-06T10:05:00Z",
            "source": "desd_product_event",
        },
    )
    updated_dataset = apply_catalog_events(dataset)
    append_order_event(
        updated_dataset,
        {
            "order_id": "O-LIVE-001",
            "customer_id": "C000001",
            "order_date": "2026-05-06",
            "items": [{"product_id": "P000020", "quantity": 7, "unit_price": 4.2}],
            "source": "desd_order_event",
        },
    )

    response = get_producer_forecast("PR000020", top_k=5, data_dir=task1_data_dir)

    assert response["producer_id"] == "PR000020"
    assert response["forecast_method"] == "latest_3_week_moving_average"
    assert response["items"][0]["product_id"] == "P000020"
    assert response["items"][0]["predicted_quantity_next_week"] == 7.0
    assert "feature_refresh_not_model_retraining" in response["limitations"]


def test_producer_forecast_rejects_unknown_producer(task1_data_dir):
    with pytest.raises(Task1DataError, match="Unknown producer_id"):
        get_producer_forecast("PR999999", data_dir=task1_data_dir)
