# Task 1 Final Evaluation Evidence

Task 1 report-ready files are generated in:

```text
outputs/task1_recommender/
```

Required evidence:

```text
recommender_metrics.csv
recommendation_examples.csv
producer_diversity.csv
product_coverage.csv
recommendation_share_by_producer.csv
producer_demand_trends.csv
producer_next_week_forecast.csv
producer_fair_reranking_alpha_study.csv
producer_fair_reranking_share_by_producer.csv
producer_fair_reranking_summary.json
discovery_metrics.csv
discovery_examples.csv
discovery_share_by_producer.csv
discovery_product_coverage.csv
task1_summary.json
```

Use these files for the intelligent ordering / quick reorder section of the report.

Live order-update integration:

```text
POST /catalog/ingest-producer
POST /catalog/ingest-product
POST /recommend/ingest-history
POST /recommend/ingest-order
outputs/logs/catalog_producer_events.jsonl
outputs/logs/catalog_product_events.jsonl
outputs/logs/recommender_history_events.jsonl
outputs/logs/recommender_order_events.jsonl
```

When DESD creates or updates a producer or product, it can send catalogue
metadata to the AI service before that product appears in order events. Customer
history can be backfilled as a batch, and each later checkout can be sent as a
single anonymised order event. The recommender overlays catalogue, history and
order events on top of the CSV seed export, so later `/recommend/reorder` calls
reflect the updated catalogue/history without direct database access.

`producer_demand_trends.csv` and `producer_next_week_forecast.csv` are the
producer-facing chart evidence. They can be used to present expected demand
direction and predicted next-week quantity by producer/product. They are
descriptive proof-of-concept trend outputs from synthetic data, not a production
forecasting model.

Producer forecast figure:

```text
docs/report_figures/research_novelty/producer_next_week_forecast_chart.png
```

The producer-fair re-ranking files are research evidence for the recommender
fairness trade-off. They compare alpha values against recommendation quality and
producer exposure concentration. They do not change the live recommender
default.

Research figure:

```text
docs/report_figures/research_novelty/producer_fair_reranking_tradeoff.png
```
