from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd

from src.recommender.schemas import DiscoveryRecommendation


METHOD_COOCCURRENCE_DISCOVERY = "co_occurrence_discovery"
METHOD_SEGMENT_POPULARITY_DISCOVERY = "segment_popularity_discovery"
METHOD_GLOBAL_POPULARITY_DISCOVERY = "global_popularity_discovery"

DISCOVERY_METHODS = {
    METHOD_COOCCURRENCE_DISCOVERY,
    METHOD_SEGMENT_POPULARITY_DISCOVERY,
    METHOD_GLOBAL_POPULARITY_DISCOVERY,
}


def discovery_reason_text(reason_codes: list[str]) -> str:
    snippets = {
        "commonly_bought_together": "commonly bought with products this customer purchased",
        "similar_basket_pattern": "found in similar baskets",
        "new_to_customer": "not previously purchased by this customer",
        "seasonally_available": "currently in season",
        "popular_with_similar_customers": "popular with similar customers",
        "globally_popular_product": "popular across the marketplace",
        "cold_start_discovery_fallback": "used as a discovery fallback where history is sparse",
        "producer_diversity_applied": "selected to improve producer diversity",
        "limited_diversity_candidates": "included because diverse alternatives were limited",
        "insufficient_new_discovery_candidates": "included because there were not enough new discovery candidates",
    }
    parts = [snippets[code] for code in reason_codes if code in snippets]
    if not parts:
        return "Recommended by the discovery scoring method."
    return "Recommended because it is " + ", ".join(parts) + "."


def build_baskets(order_lines: pd.DataFrame) -> dict[str, set[str]]:
    if order_lines.empty:
        return {}
    grouped = order_lines.groupby("order_id")["product_id"].apply(lambda values: set(map(str, values)))
    return grouped.to_dict()


def build_pairwise_cooccurrence(order_lines: pd.DataFrame) -> dict[str, Counter]:
    cooccurrence: dict[str, Counter] = defaultdict(Counter)
    for basket in build_baskets(order_lines).values():
        for left, right in combinations(sorted(basket), 2):
            cooccurrence[left][right] += 1
            cooccurrence[right][left] += 1
    return cooccurrence


def _customer_type(order_lines: pd.DataFrame, customer_id: str) -> str | None:
    rows = order_lines[order_lines["customer_id"] == customer_id]
    if rows.empty or "customer_type" not in rows:
        return None
    values = rows["customer_type"].dropna()
    return str(values.iloc[0]) if not values.empty else None


def _customer_products(order_lines: pd.DataFrame, customer_id: str) -> set[str]:
    rows = order_lines[order_lines["customer_id"] == customer_id]
    return set(map(str, rows["product_id"].dropna().unique()))


def _seasonal_score(product, recommendation_date: pd.Timestamp) -> float:
    start = int(getattr(product, "seasonal_start_month", 1) or 1)
    end = int(getattr(product, "seasonal_end_month", 12) or 12)
    month = int(recommendation_date.month)
    if start <= end:
        return 1.0 if start <= month <= end else 0.0
    return 1.0 if month >= start or month <= end else 0.0


def _catalog(products: pd.DataFrame) -> dict[str, object]:
    return {str(row.product_id): row for row in products.itertuples(index=False)}


def _segment_popularity(order_lines: pd.DataFrame, customer_type: str | None) -> dict[str, float]:
    if not customer_type:
        return {}
    rows = order_lines[order_lines["customer_type"] == customer_type]
    if rows.empty:
        return {}
    counts = rows.groupby("product_id")["quantity"].sum()
    max_count = float(counts.max()) or 1.0
    return {str(product_id): float(count) / max_count for product_id, count in counts.items()}


def _global_popularity(order_lines: pd.DataFrame) -> dict[str, float]:
    if order_lines.empty:
        return {}
    counts = order_lines.groupby("product_id")["quantity"].sum()
    max_count = float(counts.max()) or 1.0
    return {str(product_id): float(count) / max_count for product_id, count in counts.items()}


def _candidate(
    customer_id: str,
    customer_type: str | None,
    method: str,
    product,
    score: float,
    components: dict[str, float],
    based_on_product_ids: list[str],
    reason_codes: list[str],
) -> dict:
    return {
        "customer_id": customer_id,
        "customer_type": customer_type,
        "method": method,
        "product": product,
        "score": max(0.0, min(1.0, float(score))),
        "score_components": {
            key: max(0.0, min(1.0, float(value)))
            for key, value in components.items()
        },
        "based_on_product_ids": list(dict.fromkeys(map(str, based_on_product_ids))),
        "reason_codes": list(dict.fromkeys(reason_codes)),
    }


def _to_recommendation(candidate: dict, rank: int) -> DiscoveryRecommendation:
    product = candidate["product"]
    reason_codes = candidate["reason_codes"]
    return DiscoveryRecommendation(
        customer_id=candidate["customer_id"],
        customer_type=candidate["customer_type"],
        method=candidate["method"],
        rank=rank,
        product_id=str(product.product_id),
        product_name=str(product.product_name),
        producer_id=str(product.producer_id),
        score=candidate["score"],
        score_components=candidate["score_components"],
        based_on_product_ids=candidate["based_on_product_ids"],
        reason_codes=reason_codes,
        reason_text=discovery_reason_text(reason_codes),
    )


def _apply_diversity_reranking(candidates: list[dict], top_k: int) -> list[dict]:
    remaining = sorted(candidates, key=lambda item: item["score"], reverse=True)
    selected: list[dict] = []
    used_producers: set[str] = set()

    while remaining and len(selected) < top_k:
        best = remaining[0]
        alternative_index = None
        for index, candidate in enumerate(remaining):
            producer_id = str(candidate["product"].producer_id)
            if producer_id not in used_producers and candidate["score"] >= best["score"] * 0.70:
                alternative_index = index
                break

        if alternative_index is None:
            chosen = remaining.pop(0)
            if str(chosen["product"].producer_id) in used_producers:
                chosen["reason_codes"].append("limited_diversity_candidates")
        else:
            chosen = remaining.pop(alternative_index)
            if alternative_index > 0:
                chosen["reason_codes"].append("producer_diversity_applied")
                chosen["score_components"]["diversity_adjustment"] = 0.03
                chosen["score"] = min(1.0, chosen["score"] + 0.03)

        used_producers.add(str(chosen["product"].producer_id))
        selected.append(chosen)

    return selected


def _fallback_candidates(
    customer_id: str,
    customer_type: str | None,
    order_lines: pd.DataFrame,
    products: pd.DataFrame,
    purchased_products: set[str],
    recommendation_date: pd.Timestamp,
    method: str,
    exclude_purchased: bool,
) -> list[dict]:
    catalog = _catalog(products)
    segment_scores = _segment_popularity(order_lines, customer_type)
    global_scores = _global_popularity(order_lines)
    source_scores = segment_scores if method == METHOD_SEGMENT_POPULARITY_DISCOVERY and segment_scores else global_scores
    reason = (
        "popular_with_similar_customers"
        if method == METHOD_SEGMENT_POPULARITY_DISCOVERY and segment_scores
        else "globally_popular_product"
    )
    candidates = []
    for product_id, popularity in source_scores.items():
        if exclude_purchased and product_id in purchased_products:
            continue
        product = catalog.get(product_id)
        if product is None:
            continue
        seasonal = _seasonal_score(product, recommendation_date)
        reason_codes = [reason, "cold_start_discovery_fallback"]
        if product_id not in purchased_products:
            reason_codes.append("new_to_customer")
        if seasonal >= 1.0:
            reason_codes.append("seasonally_available")
        score = 0.85 * popularity + 0.15 * seasonal
        candidates.append(
            _candidate(
                customer_id,
                customer_type,
                method,
                product,
                score,
                {
                    "cooccurrence": 0.0,
                    "seasonality": 0.15 * seasonal,
                    "segment_popularity": popularity if reason == "popular_with_similar_customers" else 0.0,
                    "diversity_adjustment": 0.0,
                },
                [],
                reason_codes,
            )
        )
    return candidates


def discovery_recommendations(
    customer_id: str,
    order_lines: pd.DataFrame,
    products: pd.DataFrame,
    top_k: int = 3,
    recommendation_date: pd.Timestamp | None = None,
    method: str = METHOD_COOCCURRENCE_DISCOVERY,
) -> list[DiscoveryRecommendation]:
    if method not in DISCOVERY_METHODS:
        raise ValueError(f"Unsupported discovery method: {method}")

    top_k = max(1, int(top_k))
    recommendation_date = recommendation_date or (
        order_lines["order_date"].max() + pd.Timedelta(days=1)
    )
    customer_type = _customer_type(order_lines, customer_id)
    purchased_products = _customer_products(order_lines, customer_id)

    if method in {METHOD_SEGMENT_POPULARITY_DISCOVERY, METHOD_GLOBAL_POPULARITY_DISCOVERY}:
        candidates = _fallback_candidates(
            customer_id,
            customer_type,
            order_lines,
            products,
            purchased_products,
            recommendation_date,
            method,
            exclude_purchased=True,
        )
        if len(candidates) < top_k:
            candidates.extend(
                _fallback_candidates(
                    customer_id,
                    customer_type,
                    order_lines,
                    products,
                    purchased_products,
                    recommendation_date,
                    method,
                    exclude_purchased=False,
                )
            )
            for candidate in candidates:
                if str(candidate["product"].product_id) in purchased_products:
                    candidate["reason_codes"].append("insufficient_new_discovery_candidates")
        selected = _apply_diversity_reranking(candidates, top_k)
        return [_to_recommendation(item, rank) for rank, item in enumerate(selected, start=1)]

    cooccurrence = build_pairwise_cooccurrence(order_lines)
    catalog = _catalog(products)
    segment_scores = _segment_popularity(order_lines, customer_type)
    raw_scores: dict[str, float] = defaultdict(float)
    contributors: dict[str, Counter] = defaultdict(Counter)

    for purchased_product_id in purchased_products:
        for candidate_id, count in cooccurrence.get(purchased_product_id, {}).items():
            raw_scores[candidate_id] += float(count)
            contributors[candidate_id][purchased_product_id] += count

    max_raw = max(raw_scores.values()) if raw_scores else 0.0
    candidates = []
    for product_id, raw_score in raw_scores.items():
        if product_id in purchased_products:
            continue
        product = catalog.get(product_id)
        if product is None:
            continue
        co_score = raw_score / max_raw if max_raw else 0.0
        seasonal = _seasonal_score(product, recommendation_date)
        segment = segment_scores.get(product_id, 0.0)
        score = 0.82 * co_score + 0.10 * segment + 0.08 * seasonal
        based_on = [
            source_product_id
            for source_product_id, _ in contributors[product_id].most_common(3)
        ]
        reason_codes = ["commonly_bought_together", "similar_basket_pattern", "new_to_customer"]
        if seasonal >= 1.0:
            reason_codes.append("seasonally_available")
        if segment > 0:
            reason_codes.append("popular_with_similar_customers")
        candidates.append(
            _candidate(
                customer_id,
                customer_type,
                method,
                product,
                score,
                {
                    "cooccurrence": 0.82 * co_score,
                    "seasonality": 0.08 * seasonal,
                    "segment_popularity": 0.10 * segment,
                    "diversity_adjustment": 0.0,
                },
                based_on,
                reason_codes,
            )
        )

    if len(candidates) < top_k:
        fallback_method = (
            METHOD_SEGMENT_POPULARITY_DISCOVERY
            if customer_type
            else METHOD_GLOBAL_POPULARITY_DISCOVERY
        )
        candidates.extend(
            _fallback_candidates(
                customer_id,
                customer_type,
                order_lines,
                products,
                purchased_products,
                recommendation_date,
                fallback_method,
                exclude_purchased=True,
            )
        )
    if len(candidates) < top_k:
        candidates.extend(
            _fallback_candidates(
                customer_id,
                customer_type,
                order_lines,
                products,
                purchased_products,
                recommendation_date,
                METHOD_GLOBAL_POPULARITY_DISCOVERY,
                exclude_purchased=False,
            )
        )
        for candidate in candidates:
            if str(candidate["product"].product_id) in purchased_products:
                candidate["reason_codes"].append("insufficient_new_discovery_candidates")

    deduped = {}
    for candidate in candidates:
        product_id = str(candidate["product"].product_id)
        if product_id not in deduped or candidate["score"] > deduped[product_id]["score"]:
            deduped[product_id] = candidate

    selected = _apply_diversity_reranking(list(deduped.values()), top_k)
    return [_to_recommendation(item, rank) for rank, item in enumerate(selected, start=1)]
