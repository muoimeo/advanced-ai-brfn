from __future__ import annotations

import pandas as pd
import pytest

from src.recommender.data_loader import Task1DataError, build_order_lines, load_task1_dataset
from src.recommender.live_updates import (
    append_history_event,
    append_order_event,
    apply_history_events,
    apply_order_events,
    load_history_events,
    load_order_events,
)


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


def test_append_order_event_and_apply_overlay_updates_history(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    product = dataset.products.iloc[0]
    customer_id = str(dataset.customers.iloc[0]["customer_id"])
    event_log = tmp_path / "orders.jsonl"
    event = {
        "order_id": "O-LIVE-001",
        "customer_id": customer_id,
        "order_date": "2026-05-05",
        "items": [
            {
                "product_id": str(product["product_id"]),
                "quantity": 2,
                "unit_price": 3.5,
            }
        ],
    }

    result = append_order_event(dataset, event, path=event_log)
    updated = apply_order_events(dataset, load_order_events(event_log))
    order_lines = build_order_lines(updated)

    assert result["status"] == "ingested"
    assert result["ingested_order_lines"] == 1
    assert "O-LIVE-001" in set(updated.orders["order_id"].astype(str))
    assert "O-LIVE-001" in set(order_lines["order_id"].astype(str))
    assert pd.to_datetime(updated.orders["order_date"]).max() == pd.Timestamp("2026-05-05")


def test_append_order_event_rejects_duplicate_order_id(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    product = dataset.products.iloc[0]
    customer_id = str(dataset.customers.iloc[0]["customer_id"])
    existing_order_id = str(dataset.orders.iloc[0]["order_id"])

    with pytest.raises(Task1DataError, match="duplicates"):
        append_order_event(
            dataset,
            {
                "order_id": existing_order_id,
                "customer_id": customer_id,
                "order_date": "2026-05-05",
                "items": [
                    {
                        "product_id": str(product["product_id"]),
                        "quantity": 1,
                        "unit_price": 1.0,
                    }
                ],
            },
            path=tmp_path / "orders.jsonl",
        )


def test_append_order_event_rejects_unknown_product(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    customer_id = str(dataset.customers.iloc[0]["customer_id"])

    with pytest.raises(Task1DataError, match="unknown product_id"):
        append_order_event(
            dataset,
            {
                "order_id": "O-LIVE-002",
                "customer_id": customer_id,
                "order_date": "2026-05-05",
                "items": [
                    {
                        "product_id": "P-DOES-NOT-EXIST",
                        "quantity": 1,
                        "unit_price": 1.0,
                    }
                ],
            },
            path=tmp_path / "orders.jsonl",
        )


def test_append_history_event_and_apply_overlay_updates_history(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    product = dataset.products.iloc[0]
    customer_id = str(dataset.customers.iloc[0]["customer_id"])
    history_log = tmp_path / "history.jsonl"

    result = append_history_event(
        dataset,
        {
            "customer_id": customer_id,
            "customer_type": "family",
            "postcode_area": "BS1",
            "source": "desd_history_backfill",
            "orders": [
                {
                    "order_id": "O-HIST-001",
                    "order_date": "2026-05-01",
                    "items": [
                        {
                            "product_id": str(product["product_id"]),
                            "quantity": 3,
                            "unit_price": 2.5,
                        }
                    ],
                }
            ],
        },
        path=history_log,
    )
    updated = apply_history_events(dataset, load_history_events(history_log))
    order_lines = build_order_lines(updated)

    assert result["status"] == "ingested"
    assert result["ingested_orders"] == 1
    assert result["ingested_order_lines"] == 1
    assert "O-HIST-001" in set(updated.orders["order_id"].astype(str))
    assert "O-HIST-001" in set(order_lines["order_id"].astype(str))


def test_append_history_event_rejects_duplicate_order_id(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    product = dataset.products.iloc[0]
    customer_id = str(dataset.customers.iloc[0]["customer_id"])
    existing_order_id = str(dataset.orders.iloc[0]["order_id"])

    with pytest.raises(Task1DataError, match="duplicates"):
        append_history_event(
            dataset,
            {
                "customer_id": customer_id,
                "orders": [
                    {
                        "order_id": existing_order_id,
                        "order_date": "2026-05-01",
                        "items": [
                            {
                                "product_id": str(product["product_id"]),
                                "quantity": 1,
                                "unit_price": 2.5,
                            }
                        ],
                    }
                ],
            },
            path=tmp_path / "history.jsonl",
        )


def test_append_history_event_rejects_unknown_product(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    customer_id = str(dataset.customers.iloc[0]["customer_id"])

    with pytest.raises(Task1DataError, match="unknown product_id"):
        append_history_event(
            dataset,
            {
                "customer_id": customer_id,
                "orders": [
                    {
                        "order_id": "O-HIST-002",
                        "order_date": "2026-05-01",
                        "items": [
                            {
                                "product_id": "P-DOES-NOT-EXIST",
                                "quantity": 1,
                                "unit_price": 2.5,
                            }
                        ],
                    }
                ],
            },
            path=tmp_path / "history.jsonl",
        )
