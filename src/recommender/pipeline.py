from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import TASK1_DATA_DIR, TASK1_OUTPUT_DIR
from src.recommender.catalog_updates import (
    append_producer_event,
    append_product_event,
    apply_catalog_events,
    available_products,
)
from src.recommender.data_loader import Task1DataError, build_order_lines, load_task1_dataset
from src.recommender.discovery import (
    METHOD_COOCCURRENCE_DISCOVERY,
    discovery_recommendations,
)
from src.recommender.evaluation import export_task1_outputs, producer_next_week_forecast
from src.recommender.fairness import export_producer_fair_reranking_outputs
from src.recommender.live_updates import (
    append_history_event,
    append_order_event,
    apply_history_events,
    apply_order_events,
    load_order_events,
)
from src.recommender.quick_reorder import METHOD_FREQUENCY_RECENCY, recommend


TASK1_LIMITATIONS = [
    "recommendations_based_on_synthetic_seed_data",
    "not_production_customer_behaviour",
]

FORECAST_LIMITATIONS = [
    "descriptive_trend_forecast_from_synthetic_seed_data",
    "not_production_forecasting_model",
    "feature_refresh_not_model_retraining",
]


def _live_task1_dataset(data_dir: Path = TASK1_DATA_DIR):
    dataset = apply_catalog_events(load_task1_dataset(data_dir))
    dataset = apply_history_events(dataset)
    return apply_order_events(dataset)


def get_reorder_recommendations(
    customer_id: str,
    top_k: int = 3,
    method: str = METHOD_FREQUENCY_RECENCY,
    include_discovery: bool = False,
    data_dir: Path = TASK1_DATA_DIR,
) -> dict:
    dataset = _live_task1_dataset(data_dir)
    order_lines = build_order_lines(dataset)
    recommendation_products = available_products(dataset.products)
    recommendation_date = (
        order_lines["order_date"].max() + pd.Timedelta(days=1)
        if not order_lines.empty
        else pd.Timestamp.today().normalize()
    )
    recommendations = recommend(
        customer_id=customer_id,
        order_lines=order_lines,
        products=recommendation_products,
        method=method,
        top_k=top_k,
        recommendation_date=recommendation_date,
    )
    customer_rows = dataset.customers[dataset.customers["customer_id"] == customer_id]
    if not customer_rows.empty:
        customer_type = str(customer_rows.iloc[0]["customer_type"])
        for item in recommendations:
            if item.customer_type is None:
                item.customer_type = customer_type
    quick_reorder = [item.to_api_dict() for item in recommendations]
    response = {
        "customer_id": customer_id,
        "recommendation_date": recommendation_date.date().isoformat(),
        "method": "hybrid_reorder_discovery" if include_discovery else method,
        "top_k": top_k,
        "recommendations": quick_reorder,
        "limitations": TASK1_LIMITATIONS,
    }
    if include_discovery:
        discovery = discovery_recommendations(
            customer_id=customer_id,
            order_lines=order_lines,
            products=recommendation_products,
            top_k=top_k,
            recommendation_date=recommendation_date,
            method=METHOD_COOCCURRENCE_DISCOVERY,
        )
        if not customer_rows.empty:
            customer_type = str(customer_rows.iloc[0]["customer_type"])
            for item in discovery:
                if item.customer_type is None:
                    item.customer_type = customer_type
        response["quick_reorder"] = quick_reorder
        response["you_may_also_like"] = [item.to_api_dict() for item in discovery]
    return response


def get_producer_forecast(
    producer_id: str,
    top_k: int = 5,
    data_dir: Path = TASK1_DATA_DIR,
) -> dict:
    dataset = _live_task1_dataset(data_dir)
    producer_ids = set(dataset.producers["producer_id"].astype(str))
    if producer_id not in producer_ids:
        raise Task1DataError(f"Unknown producer_id: {producer_id}")

    forecast = producer_next_week_forecast(dataset)
    producer_rows = forecast[forecast["producer_id"].astype(str) == producer_id].head(top_k)
    items = []
    for row in producer_rows.to_dict(orient="records"):
        trend = str(row["trend_direction"])
        if trend == "up":
            alert_prefix = "High demand expected"
        elif trend == "down":
            alert_prefix = "Lower demand expected"
        else:
            alert_prefix = "Stable demand expected"
        items.append(
            {
                "product_id": str(row["product_id"]),
                "product_name": str(row["product_name"]),
                "forecast_week_start": str(row["forecast_week_start"]),
                "predicted_quantity_next_week": float(row["predicted_quantity_next_week"]),
                "trend_direction": trend,
                "basis": str(row["basis"]),
                "alert_text": (
                    f"{alert_prefix} for {row['product_name']} next week "
                    "based on recent order trends."
                ),
            }
        )

    return {
        "producer_id": producer_id,
        "forecast_method": "latest_3_week_moving_average",
        "top_k": top_k,
        "items": items,
        "limitations": FORECAST_LIMITATIONS,
    }


def ingest_order_event(
    event: dict,
    data_dir: Path = TASK1_DATA_DIR,
) -> dict:
    dataset = apply_catalog_events(load_task1_dataset(data_dir))
    dataset = apply_history_events(dataset)
    result = append_order_event(dataset, event)
    updated_dataset = apply_order_events(dataset, load_order_events())
    order_lines = build_order_lines(updated_dataset)
    result.update(
        {
            "total_orders": int(updated_dataset.orders["order_id"].nunique()),
            "total_order_lines": int(len(order_lines)),
            "limitations": [
                "ingested_order_events_are_anonymised_overlay_data",
                "advanced_ai_service_does_not_access_desd_database",
            ],
        }
    )
    return result


def ingest_history_event(
    event: dict,
    data_dir: Path = TASK1_DATA_DIR,
) -> dict:
    dataset = apply_catalog_events(load_task1_dataset(data_dir))
    result = append_history_event(dataset, event)
    dataset = apply_history_events(dataset)
    updated_dataset = apply_order_events(dataset, load_order_events())
    order_lines = build_order_lines(updated_dataset)
    result.update(
        {
            "total_orders": int(updated_dataset.orders["order_id"].nunique()),
            "total_order_lines": int(len(order_lines)),
            "limitations": [
                "ingested_history_events_are_anonymised_overlay_data",
                "advanced_ai_service_does_not_access_desd_database",
            ],
        }
    )
    return result


def ingest_producer_event(
    event: dict,
    data_dir: Path = TASK1_DATA_DIR,
) -> dict:
    dataset = load_task1_dataset(data_dir)
    return append_producer_event(dataset, event)


def ingest_product_event(
    event: dict,
    data_dir: Path = TASK1_DATA_DIR,
) -> dict:
    dataset = load_task1_dataset(data_dir)
    return append_product_event(dataset, event)


def run_task1_evaluation(
    data_dir: Path = TASK1_DATA_DIR,
    output_dir: Path = TASK1_OUTPUT_DIR,
    top_k: int = 3,
) -> dict[str, str]:
    dataset = load_task1_dataset(data_dir)
    written = export_task1_outputs(dataset, output_dir, top_k=top_k)
    written.update(
        export_producer_fair_reranking_outputs(
            data_dir=data_dir,
            output_dir=output_dir,
            top_k=top_k,
        )
    )
    return written


if __name__ == "__main__":
    written_files = run_task1_evaluation()
    for filename, path in written_files.items():
        print(f"{filename}: {path}")
