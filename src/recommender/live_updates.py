from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import LOGS_DIR
from src.recommender.data_loader import Task1DataError
from src.recommender.schemas import RecommenderDataset


ORDER_EVENT_LOG = LOGS_DIR / "recommender_order_events.jsonl"
HISTORY_EVENT_LOG = LOGS_DIR / "recommender_history_events.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                events.append(json.loads(text))
    return events


def load_order_events(path: Path = ORDER_EVENT_LOG) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def load_history_events(path: Path = HISTORY_EVENT_LOG) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def flatten_history_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for history in events:
        customer_id = str(history["customer_id"])
        for order in history.get("orders", []):
            flattened.append(
                {
                    "order_id": str(order["order_id"]),
                    "customer_id": customer_id,
                    "customer_type": history.get("customer_type"),
                    "postcode_area": history.get("postcode_area"),
                    "order_date": order["order_date"],
                    "total_amount": order.get("total_amount"),
                    "items": order.get("items", []),
                    "source": history.get("source", "desd_history_backfill"),
                    "history_batch_id": history.get("history_batch_id"),
                }
            )
    return flattened


def _known_order_ids(dataset: RecommenderDataset, events: list[dict[str, Any]]) -> set[str]:
    csv_ids = set(dataset.orders["order_id"].astype(str))
    event_ids = {str(event["order_id"]) for event in events}
    return csv_ids | event_ids


def _validate_order_event(dataset: RecommenderDataset, event: dict[str, Any], events: list[dict[str, Any]]) -> None:
    order_id = str(event.get("order_id", "")).strip()
    customer_id = str(event.get("customer_id", "")).strip()
    items = event.get("items") or []

    if not order_id:
        raise Task1DataError("Order ingest event is missing order_id.")
    if order_id in _known_order_ids(dataset, events):
        raise Task1DataError(f"Order ingest event duplicates an existing order_id: {order_id}")
    if not customer_id:
        raise Task1DataError("Order ingest event is missing customer_id.")
    if not items:
        raise Task1DataError("Order ingest event must contain at least one item.")

    known_customers = set(dataset.customers["customer_id"].astype(str))
    if customer_id not in known_customers and not event.get("customer_type"):
        raise Task1DataError(
            "Unknown customer_id. Provide customer_type for new anonymised customers."
        )

    known_products = set(dataset.products["product_id"].astype(str))
    unknown_products = sorted(
        {
            str(item.get("product_id", "")).strip()
            for item in items
            if str(item.get("product_id", "")).strip() not in known_products
        }
    )
    if unknown_products:
        raise Task1DataError(
            "Order ingest event contains unknown product_id values: "
            + ", ".join(unknown_products)
        )

    try:
        pd.to_datetime(event["order_date"])
    except Exception as exc:
        raise Task1DataError("Order ingest event contains invalid order_date.") from exc


def append_order_event(
    dataset: RecommenderDataset,
    event: dict[str, Any],
    path: Path = ORDER_EVENT_LOG,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_events = load_order_events(path)
    _validate_order_event(dataset, event, existing_events)

    normalised = {
        "order_id": str(event["order_id"]),
        "customer_id": str(event["customer_id"]),
        "customer_type": event.get("customer_type"),
        "postcode_area": event.get("postcode_area"),
        "order_date": pd.to_datetime(event["order_date"]).date().isoformat(),
        "total_amount": event.get("total_amount"),
        "items": [
            {
                "product_id": str(item["product_id"]),
                "quantity": float(item["quantity"]),
                "unit_price": float(item["unit_price"]),
            }
            for item in event["items"]
        ],
        "source": event.get("source", "desd_order_event"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if normalised["total_amount"] is None:
        normalised["total_amount"] = round(
            sum(item["quantity"] * item["unit_price"] for item in normalised["items"]),
            2,
        )
    else:
        normalised["total_amount"] = float(normalised["total_amount"])

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(normalised) + "\n")

    return {
        "status": "ingested",
        "order_id": normalised["order_id"],
        "customer_id": normalised["customer_id"],
        "ingested_order_lines": len(normalised["items"]),
        "event_log": str(path),
    }


def _normalise_history_event(dataset: RecommenderDataset, event: dict[str, Any], existing_events: list[dict[str, Any]]) -> dict[str, Any]:
    customer_id = str(event.get("customer_id", "")).strip()
    orders = event.get("orders") or []
    if not customer_id:
        raise Task1DataError("History ingest event is missing customer_id.")
    if not orders:
        raise Task1DataError("History ingest event must contain at least one order.")

    known_customers = set(dataset.customers["customer_id"].astype(str))
    if customer_id not in known_customers and not event.get("customer_type"):
        raise Task1DataError(
            "Unknown customer_id. Provide customer_type for new anonymised customers."
        )

    existing_order_ids = _known_order_ids(dataset, flatten_history_events(existing_events))
    batch_order_ids: set[str] = set()
    normalised_orders: list[dict[str, Any]] = []
    for order in orders:
        order_id = str(order.get("order_id", "")).strip()
        if not order_id:
            raise Task1DataError("History ingest order is missing order_id.")
        if order_id in existing_order_ids or order_id in batch_order_ids:
            raise Task1DataError(f"History ingest event duplicates an existing order_id: {order_id}")
        batch_order_ids.add(order_id)

        order_event = {
            "order_id": order_id,
            "customer_id": customer_id,
            "customer_type": event.get("customer_type"),
            "postcode_area": event.get("postcode_area"),
            "order_date": order.get("order_date"),
            "total_amount": order.get("total_amount"),
            "items": order.get("items") or [],
            "source": event.get("source", "desd_history_backfill"),
        }
        _validate_order_event(dataset, order_event, flatten_history_events(existing_events))
        normalised_items = [
            {
                "product_id": str(item["product_id"]),
                "quantity": float(item["quantity"]),
                "unit_price": float(item["unit_price"]),
            }
            for item in order_event["items"]
        ]
        total_amount = order_event.get("total_amount")
        if total_amount is None:
            total_amount = round(
                sum(item["quantity"] * item["unit_price"] for item in normalised_items),
                2,
            )
        else:
            total_amount = float(total_amount)

        normalised_orders.append(
            {
                "order_id": order_id,
                "order_date": pd.to_datetime(order_event["order_date"]).date().isoformat(),
                "total_amount": total_amount,
                "items": normalised_items,
            }
        )

    return {
        "history_batch_id": str(event.get("history_batch_id") or f"HIST-{datetime.now().strftime('%Y%m%d%H%M%S')}"),
        "customer_id": customer_id,
        "customer_type": event.get("customer_type"),
        "postcode_area": event.get("postcode_area"),
        "orders": normalised_orders,
        "source": event.get("source", "desd_history_backfill"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_history_event(
    dataset: RecommenderDataset,
    event: dict[str, Any],
    path: Path = HISTORY_EVENT_LOG,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_events = load_history_events(path)
    normalised = _normalise_history_event(dataset, event, existing_events)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(normalised) + "\n")

    ingested_lines = sum(len(order.get("items", [])) for order in normalised["orders"])
    return {
        "status": "ingested",
        "history_batch_id": normalised["history_batch_id"],
        "customer_id": normalised["customer_id"],
        "ingested_orders": len(normalised["orders"]),
        "ingested_order_lines": ingested_lines,
        "event_log": str(path),
    }


def apply_order_events(
    dataset: RecommenderDataset,
    events: list[dict[str, Any]] | None = None,
) -> RecommenderDataset:
    selected_events = events if events is not None else load_order_events()
    if not selected_events:
        return dataset

    customers = dataset.customers.copy()
    orders = dataset.orders.copy()
    order_items = dataset.order_items.copy()

    existing_customers = set(customers["customer_id"].astype(str))
    order_rows = []
    item_rows = []
    customer_rows = []

    for event in selected_events:
        customer_id = str(event["customer_id"])
        if customer_id not in existing_customers:
            customer_rows.append(
                {
                    "customer_id": customer_id,
                    "customer_type": event.get("customer_type") or "unknown",
                    "postcode_area": event.get("postcode_area") or "unknown",
                }
            )
            existing_customers.add(customer_id)

        order_rows.append(
            {
                "order_id": str(event["order_id"]),
                "customer_id": customer_id,
                "order_date": pd.to_datetime(event["order_date"]),
                "total_amount": float(event.get("total_amount", 0.0)),
            }
        )
        for item in event.get("items", []):
            item_rows.append(
                {
                    "order_id": str(event["order_id"]),
                    "product_id": str(item["product_id"]),
                    "quantity": float(item["quantity"]),
                    "unit_price": float(item["unit_price"]),
                }
            )

    if customer_rows:
        customers = pd.concat([customers, pd.DataFrame(customer_rows)], ignore_index=True)
    if order_rows:
        orders = pd.concat([orders, pd.DataFrame(order_rows)], ignore_index=True)
    if item_rows:
        order_items = pd.concat([order_items, pd.DataFrame(item_rows)], ignore_index=True)

    return RecommenderDataset(
        customers=customers,
        producers=dataset.producers,
        products=dataset.products,
        orders=orders,
        order_items=order_items,
        data_dir=dataset.data_dir,
    )


def apply_history_events(
    dataset: RecommenderDataset,
    events: list[dict[str, Any]] | None = None,
) -> RecommenderDataset:
    selected_events = events if events is not None else load_history_events()
    if not selected_events:
        return dataset
    return apply_order_events(dataset, flatten_history_events(selected_events))
