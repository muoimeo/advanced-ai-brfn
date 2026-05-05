from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import LOGS_DIR
from src.recommender.data_loader import Task1DataError
from src.recommender.schemas import RecommenderDataset


PRODUCT_EVENT_LOG = LOGS_DIR / "catalog_product_events.jsonl"
PRODUCER_EVENT_LOG = LOGS_DIR / "catalog_producer_events.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def load_product_events(path: Path = PRODUCT_EVENT_LOG) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def load_producer_events(path: Path = PRODUCER_EVENT_LOG) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def _latest_by_id(events: list[dict[str, Any]], id_field: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        entity_id = str(event.get(id_field, "")).strip()
        if not entity_id:
            continue
        previous = latest.get(entity_id)
        if previous is None or str(event.get("created_at", "")) >= str(previous.get("created_at", "")):
            latest[entity_id] = event
    return latest


def _normalise_producer_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("event_type") != "producer_upserted":
        raise Task1DataError("Producer event_type must be producer_upserted.")

    try:
        created_at = pd.to_datetime(event["created_at"]).isoformat()
    except Exception as exc:
        raise Task1DataError("Producer event contains invalid created_at.") from exc

    producer_id = str(event["producer_id"]).strip()
    producer_name = str(event["producer_name"]).strip()
    postcode_area = str(event["postcode_area"]).strip()
    source = str(event["source"]).strip()
    if not producer_id or not producer_name or not postcode_area or not source:
        raise Task1DataError("Producer event contains empty required fields.")

    categories = event.get("categories") or []
    if not isinstance(categories, list):
        raise Task1DataError("Producer event categories must be a list.")

    return {
        "event_type": "producer_upserted",
        "producer_id": producer_id,
        "producer_name": producer_name,
        "postcode_area": postcode_area,
        "categories": [str(category).strip() for category in categories if str(category).strip()],
        "organic_certified": event.get("organic_certified"),
        "created_at": created_at,
        "source": source,
    }


def _normalise_product_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("event_type") != "product_upserted":
        raise Task1DataError("Product event_type must be product_upserted.")

    seasonal = bool(event.get("seasonal"))
    if seasonal:
        start = int(event.get("seasonal_start_month"))
        end = int(event.get("seasonal_end_month"))
        if start < 1 or start > 12 or end < 1 or end > 12:
            raise Task1DataError("seasonal_start_month and seasonal_end_month must be between 1 and 12.")
    else:
        start = 1
        end = 12

    try:
        created_at = pd.to_datetime(event["created_at"]).isoformat()
    except Exception as exc:
        raise Task1DataError("Product event contains invalid created_at.") from exc

    price = float(event["price"])
    if price < 0:
        raise Task1DataError("Product event price must be greater than or equal to 0.")

    return {
        "event_type": "product_upserted",
        "product_id": str(event["product_id"]).strip(),
        "producer_id": str(event["producer_id"]).strip(),
        "product_name": str(event["product_name"]).strip(),
        "category": str(event["category"]).strip(),
        "unit": str(event["unit"]).strip(),
        "price": price,
        "seasonal": seasonal,
        "seasonal_start_month": start,
        "seasonal_end_month": end,
        "available": bool(event["available"]),
        "created_at": created_at,
        "source": str(event["source"]).strip(),
    }


def append_producer_event(
    dataset: RecommenderDataset,
    event: dict[str, Any],
    producer_event_path: Path = PRODUCER_EVENT_LOG,
) -> dict[str, Any]:
    normalised = _normalise_producer_event(event)

    producer_event_path.parent.mkdir(parents=True, exist_ok=True)
    normalised["ingested_at"] = datetime.now().isoformat(timespec="seconds")
    with open(producer_event_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(normalised) + "\n")

    updated = apply_producer_events(
        dataset,
        producer_events=load_producer_events(producer_event_path),
    )
    return {
        "status": "ingested",
        "producer_id": normalised["producer_id"],
        "catalog_producers": int(updated.producers["producer_id"].nunique()),
        "event_log": str(producer_event_path),
        "limitations": [
            "catalogue_events_update_metadata_without_model_retraining",
            "advanced_ai_service_does_not_access_desd_database",
        ],
    }


def append_product_event(
    dataset: RecommenderDataset,
    event: dict[str, Any],
    product_event_path: Path = PRODUCT_EVENT_LOG,
    producer_event_path: Path = PRODUCER_EVENT_LOG,
) -> dict[str, Any]:
    normalised = _normalise_product_event(event)
    producer_dataset = apply_producer_events(
        dataset,
        producer_events=load_producer_events(producer_event_path),
    )
    known_producers = set(producer_dataset.producers["producer_id"].astype(str))
    if normalised["producer_id"] not in known_producers:
        raise Task1DataError(
            f"Product event references unknown producer_id: {normalised['producer_id']}"
        )

    product_event_path.parent.mkdir(parents=True, exist_ok=True)
    normalised["ingested_at"] = datetime.now().isoformat(timespec="seconds")
    with open(product_event_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(normalised) + "\n")

    updated = apply_product_events(
        producer_dataset,
        product_events=load_product_events(product_event_path),
    )
    return {
        "status": "ingested",
        "product_id": normalised["product_id"],
        "producer_id": normalised["producer_id"],
        "available": normalised["available"],
        "catalog_products": int(updated.products["product_id"].nunique()),
        "event_log": str(product_event_path),
        "limitations": [
            "catalogue_events_update_metadata_without_model_retraining",
            "advanced_ai_service_does_not_access_desd_database",
        ],
    }


def apply_producer_events(
    dataset: RecommenderDataset,
    producer_events: list[dict[str, Any]] | None = None,
) -> RecommenderDataset:
    selected_events = producer_events if producer_events is not None else load_producer_events()
    if not selected_events:
        return dataset

    producers = dataset.producers.copy()
    latest = _latest_by_id(selected_events, "producer_id")
    rows = []
    for event in latest.values():
        normalised = _normalise_producer_event(event)
        rows.append(
            {
                "producer_id": normalised["producer_id"],
                "producer_name": normalised["producer_name"],
                "postcode_area": normalised["postcode_area"],
                "catalog_event_source": normalised["source"],
                "catalog_event_created_at": normalised["created_at"],
            }
        )

    if not rows:
        return dataset

    overlay = pd.DataFrame(rows)
    producers = producers[~producers["producer_id"].astype(str).isin(overlay["producer_id"].astype(str))]
    producers = pd.concat([producers, overlay], ignore_index=True)
    return RecommenderDataset(
        customers=dataset.customers,
        producers=producers,
        products=dataset.products,
        orders=dataset.orders,
        order_items=dataset.order_items,
        data_dir=dataset.data_dir,
    )


def apply_product_events(
    dataset: RecommenderDataset,
    product_events: list[dict[str, Any]] | None = None,
) -> RecommenderDataset:
    selected_events = product_events if product_events is not None else load_product_events()
    if not selected_events:
        return dataset

    products = dataset.products.copy()
    latest = _latest_by_id(selected_events, "product_id")
    rows = []
    for event in latest.values():
        normalised = _normalise_product_event(event)
        rows.append(
            {
                "product_id": normalised["product_id"],
                "product_name": normalised["product_name"],
                "description": normalised.get("description", ""),
                "category": normalised["category"],
                "producer_id": normalised["producer_id"],
                "seasonal_start_month": normalised["seasonal_start_month"],
                "seasonal_end_month": normalised["seasonal_end_month"],
                "base_price": normalised["price"],
                "unit": normalised["unit"],
                "allergen_info": "",
                "stock_quantity": None,
                "availability_status": "available" if normalised["available"] else "unavailable",
                "catalog_event_source": normalised["source"],
                "catalog_event_created_at": normalised["created_at"],
            }
        )

    if not rows:
        return dataset

    overlay = pd.DataFrame(rows)
    products = products[~products["product_id"].astype(str).isin(overlay["product_id"].astype(str))]
    products = pd.concat([products, overlay], ignore_index=True)
    for column in ["seasonal_start_month", "seasonal_end_month"]:
        products[column] = pd.to_numeric(products[column], errors="coerce").astype("Int64")
    products["base_price"] = pd.to_numeric(products["base_price"], errors="coerce")
    return RecommenderDataset(
        customers=dataset.customers,
        producers=dataset.producers,
        products=products,
        orders=dataset.orders,
        order_items=dataset.order_items,
        data_dir=dataset.data_dir,
    )


def apply_catalog_events(
    dataset: RecommenderDataset,
    product_events: list[dict[str, Any]] | None = None,
    producer_events: list[dict[str, Any]] | None = None,
) -> RecommenderDataset:
    with_producers = apply_producer_events(dataset, producer_events)
    return apply_product_events(with_producers, product_events)


def available_products(products: pd.DataFrame) -> pd.DataFrame:
    if "availability_status" not in products.columns:
        return products
    return products[
        products["availability_status"].fillna("available").astype(str).str.lower().ne("unavailable")
    ].copy()
