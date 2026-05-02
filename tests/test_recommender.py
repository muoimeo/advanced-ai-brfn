from __future__ import annotations

import json

import pandas as pd
import pytest

from src.recommender.data_loader import (
    Task1DataError,
    build_order_lines,
    load_task1_dataset,
)
from src.recommender.evaluation import (
    evaluate_discovery_recommenders,
    evaluate_recommenders,
    export_task1_outputs,
    producer_next_week_forecast,
    producer_demand_trends,
    temporal_split,
)
from src.recommender.discovery import (
    build_baskets,
    build_pairwise_cooccurrence,
    discovery_recommendations,
)
from src.recommender.quick_reorder import (
    METHOD_FREQUENCY_RECENCY,
    global_popularity_recommendations,
    recommend,
    user_frequency_recommendations,
)


def _write_csvs(base_dir):
    pd.DataFrame(
        [
            {"customer_id": "C000001", "customer_type": "family", "postcode_area": "BS1"},
            {"customer_id": "C000002", "customer_type": "restaurant", "postcode_area": "BS2"},
            {"customer_id": "C000003", "customer_type": "community_group", "postcode_area": "BS3"},
        ]
    ).to_csv(base_dir / "customers.csv", index=False)
    pd.DataFrame(
        [
            {"producer_id": "PR000001", "producer_name": "North Farm", "postcode_area": "BS1"},
            {"producer_id": "PR000002", "producer_name": "Valley Growers", "postcode_area": "BS2"},
        ]
    ).to_csv(base_dir / "producers.csv", index=False)
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
            {
                "product_id": "P000002",
                "product_name": "Apples",
                "category": "fruit",
                "producer_id": "PR000001",
                "seasonal_start_month": 8,
                "seasonal_end_month": 11,
                "base_price": 1.8,
            },
            {
                "product_id": "P000003",
                "product_name": "Potatoes",
                "category": "vegetables",
                "producer_id": "PR000002",
                "seasonal_start_month": 1,
                "seasonal_end_month": 12,
                "base_price": 1.2,
            },
        ]
    ).to_csv(base_dir / "products.csv", index=False)
    pd.DataFrame(
        [
            {"order_id": "O000001", "customer_id": "C000001", "order_date": "2026-01-01", "total_amount": 6.8},
            {"order_id": "O000002", "customer_id": "C000001", "order_date": "2026-01-08", "total_amount": 5.0},
            {"order_id": "O000003", "customer_id": "C000002", "order_date": "2026-01-09", "total_amount": 10.0},
            {"order_id": "O000004", "customer_id": "C000003", "order_date": "2026-01-15", "total_amount": 7.4},
            {"order_id": "O000005", "customer_id": "C000001", "order_date": "2026-02-01", "total_amount": 4.3},
            {"order_id": "O000006", "customer_id": "C000002", "order_date": "2026-02-08", "total_amount": 8.7},
        ]
    ).to_csv(base_dir / "orders.csv", index=False)
    pd.DataFrame(
        [
            {"order_id": "O000001", "product_id": "P000001", "quantity": 2, "unit_price": 2.5},
            {"order_id": "O000001", "product_id": "P000002", "quantity": 1, "unit_price": 1.8},
            {"order_id": "O000002", "product_id": "P000001", "quantity": 2, "unit_price": 2.5},
            {"order_id": "O000003", "product_id": "P000003", "quantity": 6, "unit_price": 1.2},
            {"order_id": "O000004", "product_id": "P000002", "quantity": 3, "unit_price": 1.8},
            {"order_id": "O000005", "product_id": "P000001", "quantity": 1, "unit_price": 2.5},
            {"order_id": "O000006", "product_id": "P000003", "quantity": 4, "unit_price": 1.2},
        ]
    ).to_csv(base_dir / "order_items.csv", index=False)


@pytest.fixture()
def task1_data_dir(tmp_path):
    _write_csvs(tmp_path)
    return tmp_path


def test_load_task1_dataset_validates_schema(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)

    assert dataset.customers["customer_id"].nunique() == 3
    assert dataset.products["product_id"].nunique() == 3
    assert pd.api.types.is_datetime64_any_dtype(dataset.orders["order_date"])


def test_load_task1_dataset_rejects_missing_column(task1_data_dir):
    customers = pd.read_csv(task1_data_dir / "customers.csv")
    customers.drop(columns=["postcode_area"]).to_csv(task1_data_dir / "customers.csv", index=False)

    with pytest.raises(Task1DataError, match="missing required columns"):
        load_task1_dataset(task1_data_dir)


def test_load_task1_dataset_rejects_personal_customer_columns(task1_data_dir):
    customers = pd.read_csv(task1_data_dir / "customers.csv")
    customers["email"] = "customer@example.com"
    customers.to_csv(task1_data_dir / "customers.csv", index=False)

    with pytest.raises(Task1DataError, match="personal columns"):
        load_task1_dataset(task1_data_dir)


def test_temporal_split_creates_non_empty_train_and_test(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    train, test = temporal_split(build_order_lines(dataset))

    assert not train.empty
    assert not test.empty
    assert train["order_date"].max() <= test["order_date"].max()


def test_global_popularity_returns_top_k_with_reason_codes(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    recs = global_popularity_recommendations(
        "C000001",
        build_order_lines(dataset),
        dataset.products,
        top_k=2,
    )

    assert len(recs) == 2
    assert recs[0].reason_codes == ["globally_popular_product"]


def test_user_frequency_returns_customer_specific_products(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    recs = user_frequency_recommendations(
        "C000001",
        build_order_lines(dataset),
        dataset.products,
        top_k=1,
    )

    assert recs[0].product_id == "P000001"
    assert "frequently_ordered_by_customer" in recs[0].reason_codes


def test_frequency_recency_returns_scored_recommendations(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    recs = recommend(
        "C000001",
        build_order_lines(dataset),
        dataset.products,
        method=METHOD_FREQUENCY_RECENCY,
        top_k=2,
    )

    assert len(recs) == 2
    assert all(0.0 <= item.score <= 1.0 for item in recs)
    assert all(item.reason_codes for item in recs)


def test_cold_start_customer_uses_global_fallback(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    recs = recommend(
        "C999999",
        build_order_lines(dataset),
        dataset.products,
        method=METHOD_FREQUENCY_RECENCY,
        top_k=2,
    )

    assert len(recs) == 2
    assert all("cold_start_fallback" in item.reason_codes for item in recs)


def test_evaluation_metrics_and_fairness_outputs_are_valid(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    evaluated = evaluate_recommenders(dataset, top_k=3)
    metrics = evaluated["metrics"]
    share = evaluated["recommendation_share_by_producer"]

    for column in ["precision_at_3", "recall_at_3", "hit_rate_at_3"]:
        assert metrics[column].between(0, 1).all()
    assert evaluated["product_coverage"]["product_coverage"].between(0, 1).all()
    assert evaluated["producer_diversity"]["producer_diversity"].between(0, 1).all()
    assert metrics["largest_producer_recommendation_share"].between(0, 1).all()
    assert share.groupby("method")["recommendation_share"].sum().round(6).between(0.999999, 1.000001).all()


def test_basket_creation_and_pairwise_cooccurrence(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    order_lines = build_order_lines(dataset)

    baskets = build_baskets(order_lines)
    cooccurrence = build_pairwise_cooccurrence(order_lines)

    assert baskets["O000001"] == {"P000001", "P000002"}
    assert cooccurrence["P000001"]["P000002"] == 1
    assert cooccurrence["P000002"]["P000001"] == 1


def test_discovery_excludes_purchased_products_when_possible():
    products = pd.DataFrame(
        [
            {"product_id": "P1", "product_name": "Bread", "category": "bakery", "producer_id": "PR1", "seasonal_start_month": 1, "seasonal_end_month": 12},
            {"product_id": "P2", "product_name": "Jam", "category": "preserves", "producer_id": "PR2", "seasonal_start_month": 1, "seasonal_end_month": 12},
            {"product_id": "P3", "product_name": "Cream", "category": "dairy", "producer_id": "PR3", "seasonal_start_month": 1, "seasonal_end_month": 12},
        ]
    )
    order_lines = pd.DataFrame(
        [
            {"order_id": "O1", "customer_id": "C1", "customer_type": "family", "product_id": "P1", "quantity": 1, "order_date": pd.Timestamp("2026-01-01"), "category": "bakery"},
            {"order_id": "O2", "customer_id": "C2", "customer_type": "family", "product_id": "P1", "quantity": 1, "order_date": pd.Timestamp("2026-01-02"), "category": "bakery"},
            {"order_id": "O2", "customer_id": "C2", "customer_type": "family", "product_id": "P2", "quantity": 1, "order_date": pd.Timestamp("2026-01-02"), "category": "preserves"},
            {"order_id": "O3", "customer_id": "C3", "customer_type": "family", "product_id": "P1", "quantity": 1, "order_date": pd.Timestamp("2026-01-03"), "category": "bakery"},
            {"order_id": "O3", "customer_id": "C3", "customer_type": "family", "product_id": "P3", "quantity": 1, "order_date": pd.Timestamp("2026-01-03"), "category": "dairy"},
        ]
    )

    recs = discovery_recommendations("C1", order_lines, products, top_k=2)

    assert {item.product_id for item in recs} == {"P2", "P3"}
    assert all("new_to_customer" in item.reason_codes for item in recs)
    assert all(item.score_components for item in recs)
    assert all(item.based_on_product_ids for item in recs)


def test_discovery_fallback_can_include_purchased_products_when_needed(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    order_lines = build_order_lines(dataset)

    recs = discovery_recommendations("C000001", order_lines, dataset.products, top_k=3)

    assert len(recs) == 3
    assert any("insufficient_new_discovery_candidates" in item.reason_codes for item in recs)
    assert all(item.reason_codes for item in recs)


def test_discovery_metrics_are_valid(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    evaluated = evaluate_discovery_recommenders(dataset, top_k=3)
    metrics = evaluated["discovery_metrics"]

    for column in [
        "discovery_precision_at_3",
        "discovery_recall_at_3",
        "discovery_hit_rate_at_3",
        "novelty_rate",
        "product_coverage",
        "producer_diversity",
        "largest_producer_recommendation_share",
    ]:
        assert metrics[column].between(0, 1).all()
    assert "eligible_customer_count" in metrics.columns


def test_producer_demand_trends_contains_required_fields(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    trends = producer_demand_trends(dataset)

    expected = {
        "producer_id",
        "product_id",
        "product_name",
        "week_start",
        "quantity_ordered",
        "moving_average_3w",
        "trend_direction",
    }
    assert expected <= set(trends.columns)
    assert not trends.empty


def test_producer_next_week_forecast_contains_required_fields(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    forecast = producer_next_week_forecast(dataset)

    expected = {
        "producer_id",
        "product_id",
        "product_name",
        "forecast_week_start",
        "predicted_quantity_next_week",
        "basis",
        "trend_direction",
        "forecast_note",
    }
    assert expected <= set(forecast.columns)
    assert not forecast.empty


def test_export_task1_outputs_writes_report_ready_files(task1_data_dir, tmp_path):
    dataset = load_task1_dataset(task1_data_dir)
    written = export_task1_outputs(dataset, tmp_path, top_k=3)

    expected_files = {
        "recommender_metrics.csv",
        "recommendation_examples.csv",
        "producer_diversity.csv",
        "product_coverage.csv",
        "recommendation_share_by_producer.csv",
        "discovery_metrics.csv",
        "discovery_examples.csv",
        "discovery_share_by_producer.csv",
        "discovery_product_coverage.csv",
        "producer_demand_trends.csv",
        "producer_next_week_forecast.csv",
        "task1_summary.json",
    }
    assert expected_files <= set(written)
    summary = json.loads((tmp_path / "task1_summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_type"] == "synthetic_desd_seed_export"
    assert "method_metrics" in summary
