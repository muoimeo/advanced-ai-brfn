# Advanced AI BRFN AI Service

This repository contains the Advanced AI coursework project for the BRFN case study. It is an independent AI service that stays separate from the DESD Django marketplace repository.

It covers two AI workstreams:

- Task 1: transparent quick-reorder and optional discovery recommendations from DESD-style seed transaction CSVs.
- Tasks 2-4: fruit/vegetable computer-vision classification, quality grading, model interaction/refinement support, XAI evidence, and deployment API.

The same FastAPI/Docker service exposes both workstreams. DESD integrates with this repository over HTTP rather than importing AI code or model files.

Tasks 2-4 use the Kaggle dataset:

```text
muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
```

The current framing is a 28-class image classification task. Downstream inference maps the raw class prediction into business-friendly fields such as produce type, freshness status, quality grade, and recommended action.

Quality grading is implemented as a separate rule-based layer outside the CNN.
EfficientNetB0 predicts product type and healthy/rotten condition; `src/quality/`
combines that model evidence with transparent image proxy features such as
colour profile, dark-area ratio, blur, brightness, and foreground coverage.
The deployed system does not treat the classifier as a supervised Grade A/B/C model.

## Current Status

Task 1 is implemented as a proof-of-concept recommender using fake/synthetic DESD seed export data in `data/task1/desd_seed_export/`. It compares global popularity, user-frequency, frequency+recency, co-occurrence discovery, segment-popularity discovery, and global-popularity discovery methods, then exports report-ready evaluation files to `outputs/task1_recommender/`.

The research novelty layer is implemented as evaluation evidence rather than a new production model: cost-sensitive threshold optimisation, quality-score ablation, and producer-fair re-ranking analysis.

The final selected deployment candidate is:

```text
efficientnetb0_aug_oversampled_finetuned_wsl
```

Final comparison uses `data/splits_grouped` to avoid leakage from
offline-augmented variants of the same source image crossing train,
validation, and test sets. ResNet50 fine-tuned remains the main macro-F1
challenger, but EfficientNetB0 is selected for the final demo because it has a
better deployment-risk tradeoff in the grouped XAI audit.

## Repository Boundary

The Advanced AI and DESD repositories are intentionally separate.

```text
DESD repo:
  owns Django marketplace, fake operational data, UI, cart/order flows.
  exports anonymised Task 1 CSVs.
  calls this AI service over HTTP.

Advanced AI repo:
  owns recommender algorithms, CV model inference, quality grading, evaluation, FastAPI, Docker.
```

DESD integration points:

```text
GET  /recommend/reorder?customer_id=C000003&top_k=3
POST /predict
POST /feedback
GET  /monitoring/feedback-summary
GET  /health
GET  /model-info
```

## Repository Map For Assessment

The repository is organised around two assessed workstreams:

```text
Task 1 intelligent ordering / recommendations:
  data/task1/desd_seed_export/
  src/recommender/
  notebooks/task1_quick_reorder/11_quick_reorder_task1.ipynb
  outputs/task1_recommender/
  docs/task1_*.md

Tasks 2-4 computer vision / XAI / deployment:
  src/inference/
  src/quality/
  src/api/
  src/train/
  notebooks/01_eda.ipynb to 10_quality_rule_validation.ipynb
  notebooks/custom_image_test.ipynb
  outputs/final_evaluation/
  outputs/quality_rule_eval/
  outputs/xai_examples/
  docs/xai_quality_decision_notes.md
```

The existing root-level CV notebooks are kept in place to avoid breaking paths close to submission. The split is documented through `notebooks/README_TASK1.md` and `notebooks/README_TASK234.md`.

## Local Workflow

The project can be run locally for evaluation and API testing.

1. Create and activate a virtual environment.
2. Install dependencies.
3. Use `data/splits_grouped/` for final comparison and reporting.
4. The selected model artifact is stored in `models/best_model.keras`.
5. Validate external images with `notebooks/custom_image_test.ipynb`.
6. Review report-ready outputs in `docs/report_figures/`.
7. Run the local inference API.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If TensorFlow GPU support is needed, install the TensorFlow build that matches your CUDA/cuDNN environment. CPU training works, but transfer learning experiments will be slower.

## Dataset

The EDA notebook can download the dataset with `kagglehub`:

```python
kagglehub.dataset_download("muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten")
```

Alternatively, manually extract the dataset under `data/raw/` and set `LOCAL_DATASET_PATH` inside `notebooks/01_eda.ipynb`.

The full raw image dataset is excluded from version control.

## Expected Artifacts

Tracked in Git:

```text
data/task1/desd_seed_export/
models/class_names.json
models/model_metadata.json
```

External/local artifacts:

```text
models/best_model.keras
data/splits_grouped/train.csv
data/splits_grouped/val.csv
data/splits_grouped/test.csv
outputs/eda/
outputs/figures/
outputs/confusion_matrices/
outputs/xai_examples/
outputs/quality_rule_eval/
outputs/final_evaluation/
outputs/task1_recommender/
docs/report_figures/custom_image_validation.csv
docs/report_figures/custom_image_validation_summary.csv
docs/xai_quality_decision_notes.md
```

`models/best_model.keras` is the deployment alias used by `/predict` and the
Docker image. For this submission it must be the selected
`efficientnetb0_aug_oversampled_finetuned_wsl` artifact, not the earlier
baseline CNN. The Keras binary is not committed because model binaries are
ignored. Download it from the project artifact link and place it at:

```text
models/best_model.keras
```

Project artifact link:

```text
<ADD_GOOGLE_DRIVE_OR_RELEASE_LINK_FOR_best_model.keras>
```

Task 1 recommender APIs do not need the raw image dataset. Training notebooks
need the raw Kaggle images and split files, but API inference only needs the
final model artifact, metadata files, source code, requirements, and the tracked
Task 1 CSV seed export.

## Task 1 Quick Reorder

Task 1 supports the DESD customer-facing quick reorder feature.

It answers:

```text
For this customer, which products does the AI service suggest for fast reordering?
```

Task 1 exposes two separate recommendation sections when discovery is requested:

```text
quick_reorder:
  frequency + recency recommendations for products the customer is likely to reorder

you_may_also_like:
  market-basket co-occurrence recommendations for new products the customer may like
```

Task 1 consumes normalized CSVs exported from DESD:

```text
data/task1/desd_seed_export/customers.csv
data/task1/desd_seed_export/producers.csv
data/task1/desd_seed_export/products.csv
data/task1/desd_seed_export/orders.csv
data/task1/desd_seed_export/order_items.csv
```

Current seed export scale:

```text
customers: 80
producers: 12
products: 60
orders: 483
order lines: 1,794
date range: 2026-01-10 to 2026-05-01
```

Implemented methods and business purpose:

```text
quick reorder:
  global_popularity
  user_frequency
  frequency_recency

you may also like / discovery:
  co_occurrence_discovery
  segment_popularity_discovery
  global_popularity_discovery
```

The selected default method is `frequency_recency`, which combines customer frequency, recency, customer-type affinity, and seasonality. Each recommendation includes reason codes such as:

```text
frequently_ordered_by_customer
ordered_recently
common_for_customer_type
seasonally_available
cold_start_fallback
globally_popular_product
```

The optional `you_may_also_like` branch uses pairwise market-basket co-occurrence and recommends new products where possible. It is evaluated against future unseen products and returns transparent fields such as `based_on_product_ids`, `score_components`, and discovery-specific reason codes.

Fairness and producer-bias controls are included because a recommender can over-promote already popular producers. The Task 1 evidence exports include product coverage, producer diversity, recommendation share by producer, largest producer recommendation share, and diversity-aware discovery re-ranking.

Producer-facing planning evidence is also exported:

```text
producer_demand_trends.csv
producer_next_week_forecast.csv
```

These files are descriptive proof-of-concept demand trend outputs from synthetic order history. They are not claimed as a production forecasting model.
They support the case-study producer-facing requirement by providing forecast-chart evidence such as product, producer, next forecast week, predicted quantity, trend direction, and a note explaining that the values are based on recent synthetic order trends rather than a production forecasting model.

Run the Task 1 pipeline:

```bash
python -m src.recommender.pipeline
```

Or run the orchestration notebook:

```text
notebooks/task1_quick_reorder/11_quick_reorder_task1.ipynb
```

Report-ready outputs:

```text
outputs/task1_recommender/recommender_metrics.csv
outputs/task1_recommender/recommendation_examples.csv
outputs/task1_recommender/product_coverage.csv
outputs/task1_recommender/producer_diversity.csv
outputs/task1_recommender/recommendation_share_by_producer.csv
outputs/task1_recommender/discovery_metrics.csv
outputs/task1_recommender/discovery_examples.csv
outputs/task1_recommender/discovery_share_by_producer.csv
outputs/task1_recommender/discovery_product_coverage.csv
outputs/task1_recommender/producer_demand_trends.csv
outputs/task1_recommender/producer_next_week_forecast.csv
outputs/task1_recommender/producer_fair_reranking_alpha_study.csv
outputs/task1_recommender/producer_fair_reranking_share_by_producer.csv
outputs/task1_recommender/producer_fair_reranking_summary.json
outputs/task1_recommender/task1_summary.json
```

These metrics are proof-of-concept evidence only. The data is fake/synthetic, not real BRFN customer behaviour.

Task 1 API example:

```bash
curl "http://localhost:8001/recommend/reorder?customer_id=C000003&top_k=3"
```

Hybrid quick reorder + discovery example:

```bash
curl "http://localhost:8001/recommend/reorder?customer_id=C000003&top_k=3&include_discovery=true"
```

DESD displays the returned product name, producer, score, reason codes, and limitation note. Recommendation scoring remains inside this AI service.

## Tasks 2-4 Computer Vision

Tasks 2-4 support produce quality inspection, interaction logging, and explainable decision support.

They answer:

```text
What produce is shown in the image, is it healthy/rotten, and what quality action does the AI service return?
```

Task 2 is the computer-vision classifier. The CV pipeline uses the Kaggle healthy-vs-rotten fruit/vegetable dataset as a 28-class image classification task. The final model is EfficientNetB0 fine-tuned, selected using grouped-split metrics, model comparison, high-confidence error review, XAI audit, and deployment risk.

The external quality grading layer in `src/quality/` maps model evidence and image proxy features into:

```text
Grade A / B / C / Review
colour score
size proxy score
ripeness score
recommended action
inventory status
discount suggestion
manual review flag
reason codes
warnings
```

This layer is the project response to the case-study quality requirements around colour, size, ripeness, and Grade A/B/C. Important limitation: the dataset has healthy/rotten labels, not supervised Grade A/B/C labels. Grade A/B/C is therefore produced by a transparent rule-based decision layer rather than learned directly by the CNN.

Task 3 is supported through the API interaction, feedback, and monitoring loop. `POST /feedback` records producer/admin overrides, accepted AI recommendations, predicted grade, override grade, and the quality decision snapshot. `GET /monitoring/feedback-summary` joins feedback with prediction logs by `prediction_id` and reports a human-feedback accuracy proxy over time, including daily class accuracy proxy, grade accuracy proxy, override rate, and high-confidence override count. This provides the interaction evidence needed for model monitoring, audit, and future refinement. The demo does not automatically retrain models because unverified overrides can be noisy.

Task 4 is supported through explainability and FAT evidence:

```text
Fairness:
  acknowledge dataset, lighting, background, and producer-image-condition bias

Accountability:
  prediction_id, model version, feedback logs, and override records

Trust:
  Grad-CAM attention evidence, quality component scores, reason codes, warnings, and manual-review flags
```

Grad-CAM is used as report-facing attention evidence, not as segmentation and not as a grading feature.

## CV Experiment Scope

Preferred comparison set:

1. baseline custom CNN
2. MobileNetV2
3. ResNet50
4. EfficientNetB0

Report-ready evidence is maintained for grouped-split metrics, confusion matrices, weak-class analysis, high-confidence errors, Grad-CAM examples, custom-image validation, quality-rule evaluation, Docker/API evidence, and FAT limitations.

Additional novelty evidence:

```text
cost-sensitive threshold optimisation:
  outputs/final_evaluation/cost_sensitive_threshold_results.csv
  outputs/final_evaluation/cost_sensitive_threshold_summary.json

quality-score ablation:
  outputs/final_evaluation/quality_score_ablation.csv
  outputs/quality_rule_eval/quality_score_ablation.csv

producer-fair re-ranking:
  outputs/task1_recommender/producer_fair_reranking_alpha_study.csv
  outputs/task1_recommender/producer_fair_reranking_summary.json
```

These are research/evaluation layers. They do not change the selected EfficientNetB0 model or live API defaults.

## API Contract

The service exposes:

- `GET /health`
- `POST /predict`
- `POST /feedback`
- `GET /monitoring/feedback-summary`
- `GET /model-info`
- `GET /recommend/reorder?customer_id=C000012&top_k=3`
- `GET /producer/forecast?producer_id=PR000003&top_k=5`
- `POST /catalog/ingest-producer`
- `POST /catalog/ingest-product`
- `POST /recommend/ingest-history`
- `POST /recommend/ingest-order`

The DESD system can call this API for image inference and quick reorder suggestions. See
`docs/api_contract.md` for the response schema and integration pattern.

The preferred response contract is nested:

- `prediction`: model class evidence
- `quality`: rule-based grade/action decision
- `xai`: optional attention reference, not used for grading
- `model_info`: selected model metadata

See `docs/xai_quality_decision_notes.md` for the boundary between Grad-CAM
attention evidence and the external quality grading layer.

DESD usage by task:

```text
Task 1:
  DESD sends customer_id to /recommend/reorder.
  AI returns ranked product suggestions with producer IDs, scores, reason codes, and limitations.
  DESD sends new producer metadata to /catalog/ingest-producer before products reference that producer.
  DESD sends new product metadata to /catalog/ingest-product before order events reference that product.
  DESD sends anonymised customer history to /recommend/ingest-history for initial sync or event-store recovery.
  DESD sends anonymised checkout events to /recommend/ingest-order so later recommendations include new order history.
  DESD can call /producer/forecast for producer-facing next-week demand trend cards.

Task 2:
  DESD uploads produce image to /predict.
  AI returns class prediction, healthy/rotten status, quality decision, action, warnings, and manual-review flag.

Task 3:
  DESD sends producer/admin overrides to /feedback.
  AI service logs interaction data and exposes /monitoring/feedback-summary for accuracy-proxy monitoring over time.

Task 4:
  DESD displays reason codes, quality component scores, warnings, and manual-review flags as transparent decision support.
```

## Run API Locally

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

Useful checks:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/model-info
curl "http://localhost:8001/recommend/reorder?customer_id=C000003&top_k=3"
curl "http://localhost:8001/producer/forecast?producer_id=PR000003&top_k=5"
curl "http://localhost:8001/monitoring/feedback-summary"
pytest
```

Task 1 event logs used by the live recommender:

```text
outputs/logs/catalog_producer_events.jsonl
outputs/logs/catalog_product_events.jsonl
outputs/logs/recommender_history_events.jsonl
outputs/logs/recommender_order_events.jsonl
```

Demo request examples are in `demo/demo_requests.http` and
`demo/curl_examples.md`.

## Run API With Docker

The Docker image is for CPU inference and includes the Task 1 CSV folder when present. It expects the final CV model artifacts:

```text
models/best_model.keras
models/class_names.json
models/model_metadata.json
data/task1/desd_seed_export/
```

Build and run:

```bash
docker build -t brfn-ai-service .
docker run --rm -p 8001:8001 brfn-ai-service
```

With Compose:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8001/health
http://localhost:8001/model-info
http://localhost:8001/recommend/reorder?customer_id=C000003&top_k=3
http://localhost:8001/monitoring/feedback-summary
```

## Final Evaluation Pack

After rerunning the final custom-image and quality-rule notebooks, generate the
report-ready evidence pack with:

```bash
python -m src.evaluation.build_final_evaluation_pack
```

The output is written to `outputs/final_evaluation/` and includes final model
metrics, model comparison, weak-class summary, external-image validation,
quality-rule summaries, risky cases, and an adoption recommendation.

For Task 1, run:

```bash
python -m src.recommender.pipeline
```

This writes the recommender evaluation pack to `outputs/task1_recommender/`.

For feedback-based monitoring evidence, run:

```bash
python -m src.monitoring.feedback_monitoring
```

This writes:

```text
outputs/final_evaluation/feedback_monitoring_summary.json
outputs/final_evaluation/feedback_accuracy_over_time.csv
```

These files report a human-feedback accuracy proxy, not controlled test-set accuracy. They are useful for explaining how accepted recommendations and overrides would be monitored over time after deployment.

For the three novelty evidence layers, run:

```bash
python -m src.evaluation.cost_sensitive_thresholds
python -m src.quality.evaluation
python -m src.recommender.pipeline
```

## Generative AI Usage

Generative AI was used as an engineering support tool during the project. The usage was limited to support activities rather than replacing evaluation or human technical judgement.

Uses included:

```text
synthetic data assumptions and Task 1 data-contract planning
code scaffolding for tests, API payloads, and documentation
debugging notebook/API/Docker issues
report structure and critical evaluation wording
```

Controls applied:

```text
generated code was reviewed before being kept
claims were checked against notebook outputs, API responses, and tests
synthetic data assumptions were labelled as fake/seed data
model and recommender limitations were kept explicit in docs and API responses
```

Evidence of human verification:

```text
pytest test suite
Task 1 recommender output CSVs
Task 2-4 final evaluation pack
API curl examples
Docker run instructions
```
