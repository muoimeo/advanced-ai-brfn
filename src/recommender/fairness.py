from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import TASK1_DATA_DIR, TASK1_OUTPUT_DIR
from src.recommender.data_loader import build_order_lines, load_task1_dataset
from src.recommender.discovery import (
    METHOD_COOCCURRENCE_DISCOVERY,
    discovery_recommendations,
)
from src.recommender.evaluation import _test_products_by_customer, temporal_split
from src.recommender.quick_reorder import METHOD_FREQUENCY_RECENCY, recommend
from src.recommender.schemas import RecommenderDataset


DEFAULT_ALPHA_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]


def producer_exposure_share(recommendations: list) -> dict[str, float]:
    if not recommendations:
        return {}
    counts: dict[str, int] = {}
    for item in recommendations:
        producer_id = str(item.producer_id)
        counts[producer_id] = counts.get(producer_id, 0) + 1
    total = float(sum(counts.values()))
    return {producer_id: count / total for producer_id, count in counts.items()}


def fair_rerank(recommendations: list, alpha: float, top_k: int) -> list:
    if alpha == 0:
        return list(recommendations[:top_k])

    exposure = producer_exposure_share(recommendations)
    ranked = sorted(
        recommendations,
        key=lambda item: (
            float(item.score) - alpha * exposure.get(str(item.producer_id), 0.0),
            float(item.score),
        ),
        reverse=True,
    )
    selected = list(ranked[:top_k])
    for rank, item in enumerate(selected, start=1):
        item.rank = rank
        item.score = max(0.0, min(1.0, float(item.score) - alpha * exposure.get(str(item.producer_id), 0.0)))
        if alpha > 0 and "producer_fair_reranking_applied" not in item.reason_codes:
            item.reason_codes.append("producer_fair_reranking_applied")
    return selected


def _metrics_from_frames(
    rows: list[dict],
    total_products: int,
    total_producers: int,
    precision_values: list[float],
    recall_values: list[float],
    hit_values: list[float],
    baseline_hit_rate: float | None,
) -> dict:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "precision_at_3": 0.0,
            "recall_at_3": 0.0,
            "hit_rate_at_3": 0.0,
            "product_coverage": 0.0,
            "producer_diversity": 0.0,
            "largest_producer_recommendation_share": 0.0,
            "mean_producer_exposure_share": 0.0,
            "recommendation_quality_delta": 0.0,
        }

    producer_share = frame.groupby("producer_id").size() / len(frame)
    hit_rate = sum(hit_values) / len(hit_values) if hit_values else 0.0
    baseline = hit_rate if baseline_hit_rate is None else baseline_hit_rate
    return {
        "precision_at_3": sum(precision_values) / len(precision_values) if precision_values else 0.0,
        "recall_at_3": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        "hit_rate_at_3": hit_rate,
        "product_coverage": frame["product_id"].nunique() / total_products if total_products else 0.0,
        "producer_diversity": frame["producer_id"].nunique() / total_producers if total_producers else 0.0,
        "largest_producer_recommendation_share": float(producer_share.max()),
        "mean_producer_exposure_share": float(producer_share.mean()),
        "recommendation_quality_delta": hit_rate - baseline,
    }


def evaluate_producer_fair_reranking(
    dataset: RecommenderDataset,
    top_k: int = 3,
    alpha_values: list[float] | None = None,
) -> dict[str, pd.DataFrame | dict]:
    alpha_values = alpha_values or DEFAULT_ALPHA_VALUES
    order_lines = build_order_lines(dataset)
    train, test = temporal_split(order_lines)
    test_by_customer = _test_products_by_customer(test)
    train_products_by_customer = _test_products_by_customer(train)
    recommendation_date = order_lines["order_date"].max() + pd.Timedelta(days=1)
    total_products = dataset.products["product_id"].nunique()
    total_producers = dataset.producers["producer_id"].nunique()

    metrics_rows = []
    share_frames = []
    baseline_hit_rates: dict[str, float] = {}

    for branch in ["quick_reorder", "discovery"]:
        for alpha in alpha_values:
            rows = []
            precision_values = []
            recall_values = []
            hit_values = []

            for customer_id, actual_products in test_by_customer.items():
                if branch == "quick_reorder":
                    candidates = recommend(
                        customer_id,
                        train,
                        dataset.products,
                        method=METHOD_FREQUENCY_RECENCY,
                        top_k=max(top_k * 4, top_k),
                        recommendation_date=recommendation_date,
                    )
                    target_products = actual_products
                else:
                    train_products = train_products_by_customer.get(customer_id, set())
                    target_products = actual_products - train_products
                    if not target_products:
                        continue
                    candidates = discovery_recommendations(
                        customer_id,
                        train,
                        dataset.products,
                        top_k=max(top_k * 4, top_k),
                        recommendation_date=recommendation_date,
                        method=METHOD_COOCCURRENCE_DISCOVERY,
                    )

                selected = fair_rerank(candidates, alpha=alpha, top_k=top_k)
                predicted_products = {item.product_id for item in selected}
                hits = len(predicted_products & target_products)
                precision_values.append(hits / top_k if top_k else 0.0)
                recall_values.append(hits / len(target_products) if target_products else 0.0)
                hit_values.append(1.0 if hits > 0 else 0.0)
                rows.extend(
                    {
                        "branch": branch,
                        "alpha": alpha,
                        "customer_id": item.customer_id,
                        "rank": item.rank,
                        "product_id": item.product_id,
                        "producer_id": item.producer_id,
                        "score": item.score,
                    }
                    for item in selected
                )

            baseline_hit_rate = baseline_hit_rates.get(branch)
            metrics = _metrics_from_frames(
                rows,
                total_products,
                total_producers,
                precision_values,
                recall_values,
                hit_values,
                baseline_hit_rate,
            )
            if alpha == 0:
                baseline_hit_rates[branch] = metrics["hit_rate_at_3"]
                metrics["recommendation_quality_delta"] = 0.0
            metrics_rows.append({"branch": branch, "alpha": alpha, **metrics})

            frame = pd.DataFrame(rows)
            if not frame.empty:
                share = (
                    frame.groupby(["branch", "alpha", "producer_id"])
                    .size()
                    .reset_index(name="recommendation_count")
                )
                share["recommendation_share"] = share.groupby(["branch", "alpha"])[
                    "recommendation_count"
                ].transform(lambda values: values / values.sum())
                share_frames.append(share)

    metrics_df = pd.DataFrame(metrics_rows)
    share_df = pd.concat(share_frames, ignore_index=True) if share_frames else pd.DataFrame()
    selected_rows = []
    for branch, group in metrics_df.groupby("branch"):
        baseline_largest = float(group[group["alpha"] == 0]["largest_producer_recommendation_share"].iloc[0])
        eligible = group[
            (group["largest_producer_recommendation_share"] < baseline_largest)
            & (group["recommendation_quality_delta"] >= -0.03)
        ].sort_values("alpha")
        selected_rows.append((eligible.iloc[0] if not eligible.empty else group[group["alpha"] == 0].iloc[0]).to_dict())

    return {
        "alpha_study": metrics_df,
        "share_by_producer": share_df,
        "selected": {
            row["branch"]: row for row in selected_rows
        },
    }


def export_producer_fair_reranking_outputs(
    data_dir: Path = TASK1_DATA_DIR,
    output_dir: Path = TASK1_OUTPUT_DIR,
    top_k: int = 3,
) -> dict[str, str]:
    dataset = load_task1_dataset(data_dir)
    evaluated = evaluate_producer_fair_reranking(dataset, top_k=top_k)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_path = output_dir / "producer_fair_reranking_alpha_study.csv"
    share_path = output_dir / "producer_fair_reranking_share_by_producer.csv"
    summary_path = output_dir / "producer_fair_reranking_summary.json"

    evaluated["alpha_study"].to_csv(alpha_path, index=False)
    evaluated["share_by_producer"].to_csv(share_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "selection_rule": (
                    "smallest alpha reducing largest producer share without "
                    "HitRate@3 drop greater than 0.03"
                ),
                "selected": evaluated["selected"],
                "limitations": [
                    "fair re-ranking is evaluated as research evidence",
                    "live recommender defaults are unchanged",
                    "synthetic transaction data is not production customer behaviour",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "producer_fair_reranking_alpha_study.csv": str(alpha_path),
        "producer_fair_reranking_share_by_producer.csv": str(share_path),
        "producer_fair_reranking_summary.json": str(summary_path),
    }


if __name__ == "__main__":
    written = export_producer_fair_reranking_outputs()
    for filename, path in written.items():
        print(f"{filename}: {path}")
