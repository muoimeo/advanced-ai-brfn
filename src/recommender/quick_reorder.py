from __future__ import annotations

from math import exp

import pandas as pd

from src.recommender.schemas import Recommendation


METHOD_GLOBAL_POPULARITY = "global_popularity"
METHOD_USER_FREQUENCY = "user_frequency"
METHOD_FREQUENCY_RECENCY = "frequency_recency"
SUPPORTED_METHODS = {
    METHOD_GLOBAL_POPULARITY,
    METHOD_USER_FREQUENCY,
    METHOD_FREQUENCY_RECENCY,
}


def reason_text(reason_codes: list[str]) -> str:
    snippets = {
        "globally_popular_product": "popular across the marketplace",
        "cold_start_fallback": "used as a fallback where customer history is sparse",
        "frequently_ordered_by_customer": "ordered repeatedly by this customer",
        "ordered_recently": "ordered recently by this customer",
        "common_for_customer_type": "common for this customer segment",
        "seasonally_available": "currently in season",
    }
    parts = [snippets[code] for code in reason_codes if code in snippets]
    if not parts:
        return "Recommended by the quick-reorder scoring method."
    return "Recommended because it is " + ", ".join(parts) + "."


def _customer_type(order_lines: pd.DataFrame, customer_id: str) -> str | None:
    rows = order_lines[order_lines["customer_id"] == customer_id]
    if rows.empty or "customer_type" not in rows:
        return None
    value = rows["customer_type"].dropna()
    return str(value.iloc[0]) if not value.empty else None


def _seasonal_score(product, recommendation_date: pd.Timestamp) -> float:
    start = int(getattr(product, "seasonal_start_month", 1) or 1)
    end = int(getattr(product, "seasonal_end_month", 12) or 12)
    month = int(recommendation_date.month)
    if start <= end:
        return 1.0 if start <= month <= end else 0.0
    return 1.0 if month >= start or month <= end else 0.0


def _catalog(products: pd.DataFrame) -> dict[str, object]:
    return {str(row.product_id): row for row in products.itertuples(index=False)}


def _recommendation(
    customer_id: str,
    customer_type: str | None,
    method: str,
    rank: int,
    product,
    score: float,
    reason_codes: list[str],
) -> Recommendation:
    return Recommendation(
        customer_id=customer_id,
        customer_type=customer_type,
        method=method,
        rank=rank,
        product_id=str(product.product_id),
        product_name=str(product.product_name),
        producer_id=str(product.producer_id),
        score=max(0.0, min(1.0, float(score))),
        reason_codes=list(dict.fromkeys(reason_codes)),
        reason_text=reason_text(reason_codes),
    )


def global_popularity_recommendations(
    customer_id: str,
    order_lines: pd.DataFrame,
    products: pd.DataFrame,
    top_k: int = 3,
    method: str = METHOD_GLOBAL_POPULARITY,
    include_cold_start_reason: bool = False,
) -> list[Recommendation]:
    top_k = max(1, int(top_k))
    customer_type = _customer_type(order_lines, customer_id)
    counts = (
        order_lines.groupby("product_id")["quantity"]
        .sum()
        .sort_values(ascending=False)
    )
    if counts.empty:
        return []
    max_count = float(counts.max())
    catalog = _catalog(products)
    recommendations = []
    for rank, (product_id, quantity) in enumerate(counts.head(top_k).items(), start=1):
        product = catalog.get(str(product_id))
        if product is None:
            continue
        reason_codes = ["globally_popular_product"]
        if include_cold_start_reason:
            reason_codes.append("cold_start_fallback")
        recommendations.append(
            _recommendation(
                customer_id,
                customer_type,
                method,
                rank,
                product,
                float(quantity) / max_count if max_count else 0.0,
                reason_codes,
            )
        )
    return recommendations


def user_frequency_recommendations(
    customer_id: str,
    order_lines: pd.DataFrame,
    products: pd.DataFrame,
    top_k: int = 3,
) -> list[Recommendation]:
    customer_lines = order_lines[order_lines["customer_id"] == customer_id]
    if customer_lines.empty:
        return global_popularity_recommendations(
            customer_id,
            order_lines,
            products,
            top_k,
            method=METHOD_USER_FREQUENCY,
            include_cold_start_reason=True,
        )

    customer_type = _customer_type(order_lines, customer_id)
    counts = (
        customer_lines.groupby("product_id")["quantity"]
        .sum()
        .sort_values(ascending=False)
    )
    max_count = float(counts.max())
    catalog = _catalog(products)
    recommendations = []
    for rank, (product_id, quantity) in enumerate(counts.head(top_k).items(), start=1):
        product = catalog.get(str(product_id))
        if product is None:
            continue
        recommendations.append(
            _recommendation(
                customer_id,
                customer_type,
                METHOD_USER_FREQUENCY,
                rank,
                product,
                float(quantity) / max_count if max_count else 0.0,
                ["frequently_ordered_by_customer"],
            )
        )
    return recommendations


def frequency_recency_recommendations(
    customer_id: str,
    order_lines: pd.DataFrame,
    products: pd.DataFrame,
    top_k: int = 3,
    recommendation_date: pd.Timestamp | None = None,
) -> list[Recommendation]:
    top_k = max(1, int(top_k))
    recommendation_date = recommendation_date or (
        order_lines["order_date"].max() + pd.Timedelta(days=1)
    )
    customer_lines = order_lines[order_lines["customer_id"] == customer_id]
    if customer_lines.empty:
        return global_popularity_recommendations(
            customer_id,
            order_lines,
            products,
            top_k,
            method=METHOD_FREQUENCY_RECENCY,
            include_cold_start_reason=True,
        )

    customer_type = _customer_type(order_lines, customer_id)
    grouped = customer_lines.groupby("product_id").agg(
        quantity_count=("quantity", "sum"),
        last_order_date=("order_date", "max"),
    )
    max_frequency = float(grouped["quantity_count"].max()) or 1.0

    type_lines = order_lines[order_lines["customer_type"] == customer_type]
    category_affinity = {}
    if not type_lines.empty:
        category_counts = type_lines.groupby("category")["quantity"].sum()
        max_category_count = float(category_counts.max()) or 1.0
        category_affinity = {
            str(category): float(count) / max_category_count
            for category, count in category_counts.items()
        }

    catalog = _catalog(products)
    scored = []
    for product_id, row in grouped.iterrows():
        product = catalog.get(str(product_id))
        if product is None:
            continue
        frequency_score = float(row["quantity_count"]) / max_frequency
        days_since = max(
            0,
            int((recommendation_date - row["last_order_date"]).days),
        )
        recency_score = exp(-days_since / 30.0)
        affinity_score = category_affinity.get(str(product.category), 0.0)
        seasonal_score = _seasonal_score(product, recommendation_date)
        score = (
            0.55 * frequency_score
            + 0.30 * recency_score
            + 0.10 * affinity_score
            + 0.05 * seasonal_score
        )
        reason_codes = ["frequently_ordered_by_customer"]
        if recency_score >= 0.35:
            reason_codes.append("ordered_recently")
        if affinity_score >= 0.5:
            reason_codes.append("common_for_customer_type")
        if seasonal_score >= 1.0:
            reason_codes.append("seasonally_available")
        scored.append((score, product, reason_codes))

    scored.sort(key=lambda item: item[0], reverse=True)
    recommendations = []
    for rank, (score, product, reason_codes) in enumerate(scored[:top_k], start=1):
        recommendations.append(
            _recommendation(
                customer_id,
                customer_type,
                METHOD_FREQUENCY_RECENCY,
                rank,
                product,
                score,
                reason_codes,
            )
        )
    return recommendations


def recommend(
    customer_id: str,
    order_lines: pd.DataFrame,
    products: pd.DataFrame,
    method: str = METHOD_FREQUENCY_RECENCY,
    top_k: int = 3,
    recommendation_date: pd.Timestamp | None = None,
) -> list[Recommendation]:
    if method == METHOD_GLOBAL_POPULARITY:
        return global_popularity_recommendations(customer_id, order_lines, products, top_k)
    if method == METHOD_USER_FREQUENCY:
        return user_frequency_recommendations(customer_id, order_lines, products, top_k)
    if method == METHOD_FREQUENCY_RECENCY:
        return frequency_recency_recommendations(
            customer_id,
            order_lines,
            products,
            top_k,
            recommendation_date,
        )
    raise ValueError(f"Unsupported recommender method: {method}")
