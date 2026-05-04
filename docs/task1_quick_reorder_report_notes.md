# Task 1 Report Notes

Position Task 1 as a proof-of-concept quick-reorder recommender.

Recommended wording:

```text
Because no real BRFN transaction history was available, Task 1 was implemented using fake DESD seed data exported as CSV. The recommender was evaluated with a temporal split and compared global popularity, user-frequency, and frequency-recency methods. The selected method favours transparency and integration readiness over unnecessary model complexity.
```

Why no deep recommender:

```text
A deep recommender was considered but rejected because the DESD-compatible dataset is synthetic and relatively small. Public grocery datasets provide scale but do not preserve DESD product, producer, and customer identifiers. A transparent frequency-recency method is more appropriate for an early-stage marketplace prototype.
```

Report evidence to cite:

```text
outputs/task1_recommender/recommender_metrics.csv
outputs/task1_recommender/recommendation_examples.csv
outputs/task1_recommender/product_coverage.csv
outputs/task1_recommender/producer_diversity.csv
outputs/task1_recommender/recommendation_share_by_producer.csv
outputs/task1_recommender/producer_demand_trends.csv
outputs/task1_recommender/producer_next_week_forecast.csv
outputs/task1_recommender/discovery_metrics.csv
outputs/task1_recommender/discovery_examples.csv
outputs/task1_recommender/discovery_share_by_producer.csv
outputs/task1_recommender/discovery_product_coverage.csv
outputs/task1_recommender/producer_fair_reranking_alpha_study.csv
outputs/task1_recommender/producer_fair_reranking_share_by_producer.csv
outputs/task1_recommender/producer_fair_reranking_summary.json
outputs/task1_recommender/task1_summary.json
```

Example demo evidence:

```text
docs/task1_demo_customer_C000003.md
```

Quick reorder and discovery are reported as separate recommendation tasks:

```text
quick_reorder:
  core case-study requirement; predicts likely repeat purchases

you_may_also_like:
  optional UX enhancement; suggests new products from market-basket co-occurrence
```

Discovery evaluation target:

```text
future unseen products = products purchased in the test period that were not purchased by the same customer in the train period
```

The report keeps quick reorder and discovery as separate lists so the business
purpose, metrics, and limitations remain clear.

Transparency:

```text
Task 1 does not use SHAP/LIME because the selected recommender is intentionally transparent. Each recommendation carries reason codes such as frequently ordered, ordered recently, common for customer type, seasonal availability, or cold-start fallback.
```

Research novelty / fairness trade-off:

```text
Producer-fair re-ranking is evaluated as a research layer rather than enabled by default. The alpha study compares recommendation quality against producer concentration using Precision@3, Recall@3, HitRate@3, producer diversity, product coverage and largest producer recommendation share. This formalises the case-study concern that a recommender can over-promote already visible producers.
```
