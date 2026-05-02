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
outputs/task1_recommender/task1_summary.json
```

Transparency:

```text
Task 1 does not use SHAP/LIME because the selected recommender is intentionally transparent. Each recommendation carries reason codes such as frequently ordered, ordered recently, common for customer type, seasonal availability, or cold-start fallback.
```
