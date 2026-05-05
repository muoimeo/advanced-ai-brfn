from __future__ import annotations

import json

import pandas as pd
import pytest

from src.recommender.catalog_updates import (
    append_producer_event,
    append_product_event,
    apply_catalog_events,
    apply_producer_events,
    available_products,
    load_product_events,
    load_producer_events,
)
from src.recommender.data_loader import Task1DataError, build_order_lines, load_task1_dataset
from src.recommender.live_updates import append_order_event
from src.recommender.quick_reorder import recommend


@pytest.fixture()
def task1_data_dir(tmp_path):
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
                "order_date": "2026-01-01",
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
    return tmp_path


def _product_event(**overrides):
    event = {
        "event_type": "product_upserted",
        "product_id": "P000002",
        "producer_id": "PR000001",
        "product_name": "Walnut Bread",
        "category": "bakery",
        "unit": "loaf",
        "price": 4.2,
        "seasonal": True,
        "seasonal_start_month": 1,
        "seasonal_end_month": 12,
        "available": True,
        "created_at": "2026-05-06T10:30:00Z",
        "source": "desd_product_event",
    }
    event.update(overrides)
    return event


def _producer_event(**overrides):
    event = {
        "event_type": "producer_upserted",
        "producer_id": "PR000020",
        "producer_name": "New Bristol Bakery",
        "postcode_area": "BS1",
        "categories": ["bakery"],
        "organic_certified": False,
        "created_at": "2026-05-06T10:00:00Z",
        "source": "desd_producer_event",
    }
    event.update(overrides)
    return event


def test_append_producer_event_accepts_canonical_schema(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    producer_log = tmp_path / "producers.jsonl"

    response = append_producer_event(dataset, _producer_event(), producer_event_path=producer_log)
    updated = apply_producer_events(dataset, producer_events=load_producer_events(producer_log))

    assert response["status"] == "ingested"
    assert response["producer_id"] == "PR000020"
    assert "PR000020" in set(updated.producers["producer_id"].astype(str))


def test_append_producer_event_rejects_invalid_event_type(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)

    with pytest.raises(Task1DataError, match="producer_upserted"):
        append_producer_event(
            dataset,
            _producer_event(event_type="producer_deleted"),
            producer_event_path=tmp_path / "producers.jsonl",
        )


def test_latest_producer_upsert_wins(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    producer_log = tmp_path / "producers.jsonl"
    append_producer_event(
        dataset,
        _producer_event(producer_name="Old Bakery", created_at="2026-05-06T10:00:00Z"),
        producer_event_path=producer_log,
    )
    append_producer_event(
        dataset,
        _producer_event(producer_name="New Bakery", created_at="2026-05-06T11:00:00Z"),
        producer_event_path=producer_log,
    )

    updated = apply_producer_events(dataset, producer_events=load_producer_events(producer_log))
    row = updated.producers[updated.producers["producer_id"] == "PR000020"].iloc[0]

    assert row["producer_name"] == "New Bakery"


def test_append_product_event_accepts_canonical_schema(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    product_log = tmp_path / "products.jsonl"

    response = append_product_event(dataset, _product_event(), product_event_path=product_log)
    updated = apply_catalog_events(dataset, product_events=load_product_events(product_log))

    assert response["status"] == "ingested"
    assert response["product_id"] == "P000002"
    assert "P000002" in set(updated.products["product_id"].astype(str))


def test_append_product_event_rejects_invalid_event_type(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)

    with pytest.raises(Task1DataError, match="product_upserted"):
        append_product_event(
            dataset,
            _product_event(event_type="product_deleted"),
            product_event_path=tmp_path / "products.jsonl",
        )


def test_append_product_event_rejects_invalid_seasonal_months(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)

    with pytest.raises(Task1DataError, match="between 1 and 12"):
        append_product_event(
            dataset,
            _product_event(seasonal_start_month=0),
            product_event_path=tmp_path / "products.jsonl",
        )


def test_non_seasonal_product_normalises_to_full_year(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    product_log = tmp_path / "products.jsonl"

    append_product_event(
        dataset,
        _product_event(seasonal=False, seasonal_start_month=None, seasonal_end_month=None),
        product_event_path=product_log,
    )
    updated = apply_catalog_events(dataset, product_events=load_product_events(product_log))
    row = updated.products[updated.products["product_id"] == "P000002"].iloc[0]

    assert int(row["seasonal_start_month"]) == 1
    assert int(row["seasonal_end_month"]) == 12


def test_append_product_event_rejects_unknown_producer(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)

    with pytest.raises(Task1DataError, match="unknown producer_id"):
        append_product_event(
            dataset,
            _product_event(producer_id="PR999999"),
            product_event_path=tmp_path / "products.jsonl",
        )


def test_append_product_event_accepts_producer_from_overlay(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    producer_log = tmp_path / "producers.jsonl"
    producer_log.write_text(
        json.dumps(
            {
                "event_type": "producer_upserted",
                "producer_id": "PR000020",
                "producer_name": "New Bristol Bakery",
                "postcode_area": "BS1",
                "created_at": "2026-05-06T10:00:00Z",
                "source": "desd_producer_event",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    response = append_product_event(
        dataset,
        _product_event(producer_id="PR000020"),
        product_event_path=tmp_path / "products.jsonl",
        producer_event_path=producer_log,
    )

    assert response["status"] == "ingested"
    assert response["producer_id"] == "PR000020"


def test_latest_product_upsert_wins(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    product_log = tmp_path / "products.jsonl"
    append_product_event(
        dataset,
        _product_event(product_name="Old Walnut Bread", created_at="2026-05-06T10:00:00Z"),
        product_event_path=product_log,
    )
    append_product_event(
        dataset,
        _product_event(product_name="New Walnut Bread", created_at="2026-05-06T11:00:00Z"),
        product_event_path=product_log,
    )

    updated = apply_catalog_events(dataset, product_events=load_product_events(product_log))
    row = updated.products[updated.products["product_id"] == "P000002"].iloc[0]

    assert row["product_name"] == "New Walnut Bread"


def test_unavailable_product_is_excluded_from_recommendation_candidates(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    product_log = tmp_path / "products.jsonl"
    append_product_event(
        dataset,
        _product_event(product_id="P000001", available=False),
        product_event_path=product_log,
    )
    updated = apply_catalog_events(dataset, product_events=load_product_events(product_log))

    recs = recommend(
        "C000001",
        build_order_lines(updated),
        available_products(updated.products),
        top_k=3,
    )

    assert all(item.product_id != "P000001" for item in recs)


def test_order_ingest_accepts_newly_ingested_product(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    product_log = tmp_path / "products.jsonl"
    append_product_event(dataset, _product_event(), product_event_path=product_log)
    updated = apply_catalog_events(dataset, product_events=load_product_events(product_log))

    result = append_order_event(
        updated,
        {
            "order_id": "O-LIVE-001",
            "customer_id": "C000001",
            "order_date": "2026-05-06",
            "items": [
                {
                    "product_id": "P000002",
                    "quantity": 1,
                    "unit_price": 4.2,
                }
            ],
        },
        path=tmp_path / "orders.jsonl",
    )

    assert result["status"] == "ingested"
