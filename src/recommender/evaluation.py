from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.recommender.data_loader import build_order_lines, dataset_summary
from src.recommender.quick_reorder import (
    METHOD_FREQUENCY_RECENCY,
    METHOD_GLOBAL_POPULARITY,
    METHOD_USER_FREQUENCY,
    recommend,
)
from src.recommender.schemas import RecommenderDataset


DEFAULT_METHODS = [
    METHOD_GLOBAL_POPULARITY,
    METHOD_USER_FREQUENCY,
    METHOD_FREQUENCY_RECENCY,
]


def temporal_split(
    order_lines: pd.DataFrame,
    train_ratio: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if order_lines.empty:
        return order_lines.copy(), order_lines.copy()

    orders = (
        order_lines[["order_id", "order_date"]]
        .drop_duplicates()
        .sort_values(["order_date", "order_id"])
    )
    split_index = int(len(orders) * train_ratio)
    split_index = min(max(split_index, 1), len(orders) - 1)
    train_order_ids = set(orders.iloc[:split_index]["order_id"])
    train = order_lines[order_lines["order_id"].isin(train_order_ids)].copy()
    test = order_lines[~order_lines["order_id"].isin(train_order_ids)].copy()
    return train, test


def _test_products_by_customer(test: pd.DataFrame) -> dict[str, set[str]]:
    grouped = test.groupby("customer_id")["product_id"].apply(lambda x: set(map(str, x)))
    return grouped.to_dict()


def _recommendations_to_frame(recommendations: list) -> pd.DataFrame:
    return pd.DataFrame([recommendation.to_dict() for recommendation in recommendations])


def evaluate_recommenders(
    dataset: RecommenderDataset,
    top_k: int = 3,
    methods: list[str] | None = None,
) -> dict[str, pd.DataFrame | dict]:
    methods = methods or DEFAULT_METHODS
    order_lines = build_order_lines(dataset)
    train, test = temporal_split(order_lines)
    test_by_customer = _test_products_by_customer(test)
    recommendation_date = order_lines["order_date"].max() + pd.Timedelta(days=1)

    metrics_rows = []
    recommendation_frames = []
    share_rows = []
    total_products = dataset.products["product_id"].nunique()
    total_producers = dataset.producers["producer_id"].nunique()

    for method in methods:
        method_recommendations = []
        precision_values = []
        recall_values = []
        hit_values = []
        cold_start_count = 0

        for customer_id, actual_products in test_by_customer.items():
            recommendations = recommend(
                customer_id=customer_id,
                order_lines=train,
                products=dataset.products,
                method=method,
                top_k=top_k,
                recommendation_date=recommendation_date,
            )
            if any("cold_start_fallback" in item.reason_codes for item in recommendations):
                cold_start_count += 1
            predicted_products = {item.product_id for item in recommendations}
            hits = len(predicted_products & actual_products)
            precision_values.append(hits / top_k if top_k else 0.0)
            recall_values.append(hits / len(actual_products) if actual_products else 0.0)
            hit_values.append(1.0 if hits > 0 else 0.0)
            method_recommendations.extend(recommendations)

        frame = _recommendations_to_frame(method_recommendations)
        if not frame.empty:
            recommendation_frames.append(frame)
            producer_counts = frame.groupby("producer_id").size().reset_index(name="recommendation_count")
            total_recs = float(producer_counts["recommendation_count"].sum())
            producer_counts["recommendation_share"] = (
                producer_counts["recommendation_count"] / total_recs if total_recs else 0.0
            )
            producer_counts.insert(0, "method", method)
            share_rows.append(producer_counts)
            unique_products = frame["product_id"].nunique()
            unique_producers = frame["producer_id"].nunique()
        else:
            unique_products = 0
            unique_producers = 0

        largest_producer_share = 0.0
        if share_rows and not producer_counts.empty:
            largest_producer_share = float(producer_counts["recommendation_share"].max())

        metrics_rows.append(
            {
                "method": method,
                "precision_at_3": sum(precision_values) / len(precision_values) if precision_values else 0.0,
                "recall_at_3": sum(recall_values) / len(recall_values) if recall_values else 0.0,
                "hit_rate_at_3": sum(hit_values) / len(hit_values) if hit_values else 0.0,
                "product_coverage": unique_products / total_products if total_products else 0.0,
                "producer_diversity": unique_producers / total_producers if total_producers else 0.0,
                "unique_recommended_producers": unique_producers,
                "total_producers": total_producers,
                "largest_producer_recommendation_share": largest_producer_share,
                "cold_start_fallback_count": cold_start_count,
                "evaluated_customers": len(test_by_customer),
                "notes": "Synthetic DESD seed data; proof-of-concept metrics only.",
            }
        )

    recommendations_df = (
        pd.concat(recommendation_frames, ignore_index=True)
        if recommendation_frames
        else pd.DataFrame()
    )
    producer_share_df = (
        pd.concat(share_rows, ignore_index=True) if share_rows else pd.DataFrame()
    )
    metrics_df = pd.DataFrame(metrics_rows)
    product_coverage_df = metrics_df[
        ["method", "product_coverage"]
    ].copy()
    product_coverage_df["unique_recommended_products"] = (
        product_coverage_df["product_coverage"] * total_products
    ).round().astype(int)
    product_coverage_df["total_products"] = int(total_products)
    product_coverage_df = product_coverage_df[
        ["method", "unique_recommended_products", "total_products", "product_coverage"]
    ]

    producer_diversity_df = metrics_df[
        ["method", "producer_diversity"]
    ].copy()
    producer_diversity_df["unique_recommended_producers"] = (
        producer_diversity_df["producer_diversity"] * total_producers
    ).round().astype(int)
    producer_diversity_df["total_producers"] = int(total_producers)
    producer_diversity_df = producer_diversity_df[
        ["method", "unique_recommended_producers", "total_producers", "producer_diversity"]
    ]

    return {
        "metrics": metrics_df,
        "recommendation_examples": recommendations_df,
        "product_coverage": product_coverage_df,
        "producer_diversity": producer_diversity_df,
        "recommendation_share_by_producer": producer_share_df,
        "dataset_summary": dataset_summary(dataset, order_lines),
    }


def producer_demand_trends(dataset: RecommenderDataset) -> pd.DataFrame:
    order_lines = build_order_lines(dataset)
    if order_lines.empty:
        return pd.DataFrame(
            columns=[
                "producer_id",
                "product_id",
                "product_name",
                "week_start",
                "quantity_ordered",
                "moving_average_3w",
                "trend_direction",
            ]
        )
    lines = order_lines.copy()
    lines["week_start"] = (
        lines["order_date"].dt.to_period("W").apply(lambda period: period.start_time.date().isoformat())
    )
    weekly = (
        lines.groupby(["producer_id", "product_id", "product_name", "week_start"])["quantity"]
        .sum()
        .reset_index(name="quantity_ordered")
        .sort_values(["producer_id", "product_id", "week_start"])
    )
    weekly["moving_average_3w"] = (
        weekly.groupby(["producer_id", "product_id"])["quantity_ordered"]
        .transform(lambda series: series.rolling(3, min_periods=1).mean())
        .round(4)
    )
    previous = weekly.groupby(["producer_id", "product_id"])["moving_average_3w"].shift(1)
    weekly["trend_direction"] = "stable"
    weekly.loc[weekly["moving_average_3w"] > previous, "trend_direction"] = "up"
    weekly.loc[weekly["moving_average_3w"] < previous, "trend_direction"] = "down"
    return weekly


def producer_next_week_forecast(dataset: RecommenderDataset) -> pd.DataFrame:
    trends = producer_demand_trends(dataset)
    columns = [
        "producer_id",
        "product_id",
        "product_name",
        "forecast_week_start",
        "predicted_quantity_next_week",
        "basis",
        "trend_direction",
        "forecast_note",
    ]
    if trends.empty:
        return pd.DataFrame(columns=columns)

    trends = trends.copy()
    trends["week_start_dt"] = pd.to_datetime(trends["week_start"])
    idx = trends.groupby(["producer_id", "product_id"])["week_start_dt"].idxmax()
    latest = trends.loc[idx].copy()
    latest["forecast_week_start"] = (
        latest["week_start_dt"] + pd.Timedelta(days=7)
    ).dt.date.astype(str)
    latest["predicted_quantity_next_week"] = latest["moving_average_3w"].round(2)
    latest["basis"] = "latest_3_week_moving_average"
    latest["forecast_note"] = (
        "Descriptive proof-of-concept forecast from synthetic seed data; "
        "not a production demand forecasting model."
    )
    return latest[columns].sort_values(
        ["producer_id", "predicted_quantity_next_week"],
        ascending=[True, False],
    )


def export_task1_outputs(
    dataset: RecommenderDataset,
    output_dir: Path,
    top_k: int = 3,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated = evaluate_recommenders(dataset, top_k=top_k)
    written = {}

    file_map = {
        "metrics": "recommender_metrics.csv",
        "recommendation_examples": "recommendation_examples.csv",
        "producer_diversity": "producer_diversity.csv",
        "product_coverage": "product_coverage.csv",
        "recommendation_share_by_producer": "recommendation_share_by_producer.csv",
    }
    for key, filename in file_map.items():
        path = output_dir / filename
        dataframe = evaluated[key]
        dataframe.to_csv(path, index=False)
        written[filename] = str(path)

    trends = producer_demand_trends(dataset)
    trends_path = output_dir / "producer_demand_trends.csv"
    trends.to_csv(trends_path, index=False)
    written["producer_demand_trends.csv"] = str(trends_path)

    forecast = producer_next_week_forecast(dataset)
    forecast_path = output_dir / "producer_next_week_forecast.csv"
    forecast.to_csv(forecast_path, index=False)
    written["producer_next_week_forecast.csv"] = str(forecast_path)

    summary = evaluated["dataset_summary"]
    method_metrics = (
        evaluated["metrics"].set_index("method").to_dict(orient="index")
        if not evaluated["metrics"].empty
        else {}
    )
    summary.update(
        {
            "selected_method": METHOD_FREQUENCY_RECENCY,
            "reason": (
                "best balance between personalisation, transparency, and "
                "producer-diversity evidence on synthetic DESD-style data"
            ),
            "top_k": top_k,
            "method_metrics": method_metrics,
        }
    )
    summary_path = output_dir / "task1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written["task1_summary.json"] = str(summary_path)
    return written
