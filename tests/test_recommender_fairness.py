import pandas as pd
import pytest

from src.recommender.fairness import (
    evaluate_producer_fair_reranking,
    fair_rerank,
)
from src.recommender.quick_reorder import recommend
from src.recommender.data_loader import build_order_lines, load_task1_dataset


@pytest.fixture()
def task1_data_dir(tmp_path):
    pd.DataFrame(
        [
            {"customer_id": "C000001", "customer_type": "family", "postcode_area": "BS1"},
            {"customer_id": "C000002", "customer_type": "restaurant", "postcode_area": "BS2"},
            {"customer_id": "C000003", "customer_type": "family", "postcode_area": "BS3"},
        ]
    ).to_csv(tmp_path / "customers.csv", index=False)
    pd.DataFrame(
        [
            {"producer_id": "PR000001", "producer_name": "North Farm", "postcode_area": "BS1"},
            {"producer_id": "PR000002", "producer_name": "Valley Growers", "postcode_area": "BS2"},
        ]
    ).to_csv(tmp_path / "producers.csv", index=False)
    pd.DataFrame(
        [
            {"product_id": "P000001", "product_name": "Tomatoes", "category": "vegetables", "producer_id": "PR000001", "seasonal_start_month": 1, "seasonal_end_month": 12, "base_price": 2.5},
            {"product_id": "P000002", "product_name": "Apples", "category": "fruit", "producer_id": "PR000001", "seasonal_start_month": 8, "seasonal_end_month": 11, "base_price": 1.8},
            {"product_id": "P000003", "product_name": "Potatoes", "category": "vegetables", "producer_id": "PR000002", "seasonal_start_month": 1, "seasonal_end_month": 12, "base_price": 1.2},
        ]
    ).to_csv(tmp_path / "products.csv", index=False)
    pd.DataFrame(
        [
            {"order_id": "O000001", "customer_id": "C000001", "order_date": "2026-01-01", "total_amount": 6.8},
            {"order_id": "O000002", "customer_id": "C000001", "order_date": "2026-01-08", "total_amount": 5.0},
            {"order_id": "O000003", "customer_id": "C000002", "order_date": "2026-01-09", "total_amount": 10.0},
            {"order_id": "O000004", "customer_id": "C000003", "order_date": "2026-01-15", "total_amount": 7.4},
            {"order_id": "O000005", "customer_id": "C000001", "order_date": "2026-02-01", "total_amount": 4.3},
            {"order_id": "O000006", "customer_id": "C000002", "order_date": "2026-02-08", "total_amount": 8.7},
        ]
    ).to_csv(tmp_path / "orders.csv", index=False)
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
    ).to_csv(tmp_path / "order_items.csv", index=False)
    return tmp_path


def test_fair_reranking_alpha_zero_returns_original_ranking(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    recs = recommend("C000001", build_order_lines(dataset), dataset.products, top_k=3)

    reranked = fair_rerank(recs, alpha=0.0, top_k=3)

    assert [item.product_id for item in reranked] == [item.product_id for item in recs]


def test_fair_reranking_preserves_top_k_length(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    recs = recommend("C000001", build_order_lines(dataset), dataset.products, top_k=3)

    reranked = fair_rerank(recs, alpha=0.2, top_k=2)

    assert len(reranked) == 2
    assert all("producer_fair_reranking_applied" in item.reason_codes for item in reranked)


def test_producer_fairness_metrics_are_bounded(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    evaluated = evaluate_producer_fair_reranking(
        dataset,
        top_k=3,
        alpha_values=[0.0, 0.1],
    )
    metrics = evaluated["alpha_study"]

    for column in [
        "precision_at_3",
        "recall_at_3",
        "hit_rate_at_3",
        "product_coverage",
        "producer_diversity",
        "largest_producer_recommendation_share",
        "mean_producer_exposure_share",
    ]:
        assert metrics[column].between(0, 1).all()
    assert set(metrics["branch"]) == {"quick_reorder", "discovery"}


def test_higher_alpha_can_reduce_or_preserve_largest_producer_share(task1_data_dir):
    dataset = load_task1_dataset(task1_data_dir)
    evaluated = evaluate_producer_fair_reranking(
        dataset,
        top_k=3,
        alpha_values=[0.0, 0.3],
    )
    metrics = evaluated["alpha_study"]

    for branch, group in metrics.groupby("branch"):
        base_share = float(group[group["alpha"] == 0.0]["largest_producer_recommendation_share"].iloc[0])
        fair_share = float(group[group["alpha"] == 0.3]["largest_producer_recommendation_share"].iloc[0])
        assert fair_share <= base_share
