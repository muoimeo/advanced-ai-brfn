# Task 1 Demo Customer Evidence: C000003

This file records one report/demo example for the Task 1 quick-reorder recommender.

Important limitation: the underlying DESD export is fake/synthetic seed data. This example demonstrates the recommender pipeline and API contract, not real customer behaviour.

## Customer Profile

```text
customer_id: C000003
customer_type: young_professional
postcode_area: BS1
```

## Purchase History Summary

Customer `C000003` has 7 synthetic historical orders between `2026-01-18` and `2026-04-12`.

Top historical products:

| product_id | product_name | producer_id | quantity | orders | last_order_date |
|---|---|---:|---:|---:|---|
| P000113 | Autumn Raspberries | PR000008 | 5 | 2 | 2026-04-01 |
| P000104 | Morning Breakfast Buns | PR000011 | 5 | 2 | 2026-02-18 |
| P000107 | Victoria Plums | PR000002 | 4 | 2 | 2026-04-12 |
| P000137 | Red Gooseberries | PR000008 | 3 | 1 | 2026-02-18 |
| P000106 | Crown Prince Squash | PR000013 | 3 | 1 | 2026-02-10 |
| P000083 | Somerset Discovery Apples | PR000002 | 3 | 1 | 2026-02-01 |

## API Request

```powershell
curl.exe "http://localhost:8001/recommend/reorder?customer_id=C000003&top_k=3"
```

Hybrid quick reorder + discovery request:

```powershell
curl.exe "http://localhost:8001/recommend/reorder?customer_id=C000003&top_k=3&include_discovery=true"
```

## API Output

```json
{
  "customer_id": "C000003",
  "recommendation_date": "2026-05-02",
  "method": "frequency_recency",
  "top_k": 3,
  "recommendations": [
    {
      "customer_id": "C000003",
      "customer_type": "young_professional",
      "method": "frequency_recency",
      "rank": 1,
      "product_id": "P000113",
      "product_name": "Autumn Raspberries",
      "producer_id": "PR000008",
      "score": 0.794796,
      "reason_codes": [
        "frequently_ordered_by_customer",
        "ordered_recently",
        "common_for_customer_type",
        "seasonally_available"
      ],
      "reason_text": "Recommended because it is ordered repeatedly by this customer, ordered recently by this customer, common for this customer segment, currently in season."
    },
    {
      "customer_id": "C000003",
      "customer_type": "young_professional",
      "method": "frequency_recency",
      "rank": 2,
      "product_id": "P000107",
      "product_name": "Victoria Plums",
      "producer_id": "PR000002",
      "score": 0.732075,
      "reason_codes": [
        "frequently_ordered_by_customer",
        "ordered_recently",
        "common_for_customer_type",
        "seasonally_available"
      ],
      "reason_text": "Recommended because it is ordered repeatedly by this customer, ordered recently by this customer, common for this customer segment, currently in season."
    },
    {
      "customer_id": "C000003",
      "customer_type": "young_professional",
      "method": "frequency_recency",
      "rank": 3,
      "product_id": "P000104",
      "product_name": "Morning Breakfast Buns",
      "producer_id": "PR000011",
      "score": 0.699279,
      "reason_codes": [
        "frequently_ordered_by_customer",
        "common_for_customer_type",
        "seasonally_available"
      ],
      "reason_text": "Recommended because it is ordered repeatedly by this customer, common for this customer segment, currently in season."
    }
  ],
  "limitations": [
    "recommendations_based_on_synthetic_seed_data",
    "not_production_customer_behaviour"
  ]
}
```

## Interpretation For Report

The recommendations are consistent with the synthetic purchase history:

- `Autumn Raspberries` was ordered twice with total quantity 5 and appears as rank 1.
- `Victoria Plums` was ordered twice and most recently on `2026-04-12`, so the recency component raises it to rank 2.
- `Morning Breakfast Buns` was also ordered twice with total quantity 5, but was last ordered earlier on `2026-02-18`, so it ranks below the more recent plum purchase.

This demonstrates why the selected frequency-recency method is more useful for quick reorder than a pure popularity baseline: it can generate customer-specific recommendations and attach transparent reason codes.

## You May Also Like Discovery Example

When `include_discovery=true`, the API also returns a separate `you_may_also_like` section. This section is not a reorder list. It suggests new products using market-basket co-occurrence.

Example discovery output for `C000003`:

| rank | product_id | product_name | producer_id | score | based_on_product_ids |
|---:|---|---|---|---:|---|
| 1 | P000096 | New Season Potatoes | PR000003 | 0.973333 | P000106, P000119, P000097 |
| 2 | P000126 | Mixed Salad Leaves | PR000009 | 0.937708 | P000106, P000097, P000107 |
| 3 | P000140 | Apple and Oat Traybake | PR000011 | 0.892083 | P000106, P000134, P000137 |

Interpretation:

- `quick_reorder` recommends products already bought repeatedly or recently.
- `you_may_also_like` excludes previously purchased products where possible and recommends products commonly found in similar baskets.
- The discovery branch includes `score_components` for transparency: co-occurrence, seasonality, segment popularity, and diversity adjustment.
- Producer diversity is applied as a re-ranking preference, not a hard rule.

Example reason codes:

```text
commonly_bought_together
similar_basket_pattern
new_to_customer
seasonally_available
popular_with_similar_customers
```

Use careful wording:

```text
The example demonstrates recommender behaviour on synthetic DESD-style data. It should not be interpreted as evidence of real customer preference or production recommendation accuracy.
```
