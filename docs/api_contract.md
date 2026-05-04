# AI Service API Contract

This repository exposes the produce freshness model as a standalone FastAPI
service. DESD should integrate with it over HTTP rather than merging the
repositories.

## Base URL

Local development:

```text
http://localhost:8001
```

Docker network example:

```text
http://ai-service:8001
```

## Endpoints

### GET /health

Checks that the API is running and whether the model can be loaded.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### GET /model-info

Returns metadata for the selected final model.

Important fields:

```json
{
  "model_name": "efficientnetb0_aug_oversampled_finetuned_wsl",
  "base_model_family": "EfficientNetB0",
  "num_classes": 28,
  "image_size": [224, 224],
  "metrics": {
    "test_accuracy": 0.9719626168224299,
    "macro_f1": 0.9612978974472393,
    "weighted_f1": 0.9718550099623252
  }
}
```

### POST /predict

Accepts one JPEG or PNG image using `multipart/form-data`.

Request field:

```text
file
```

Example response:

```json
{
  "prediction_id": "uuid",
  "model_name": "efficientnetb0_aug_oversampled_finetuned_wsl",
  "model_version": "2026-04-29T15:45:22",
  "produce_type": "Banana",
  "predicted_class": "Banana__Rotten",
  "freshness_status": "rotten",
  "confidence": 0.91,
  "top1_top2_margin": 0.86,
  "confidence_score": 0.91,
  "freshness_score": 0.09,
  "quality_grade": "C",
  "recommended_action": "manual_review_or_discard",
  "reason_codes": ["ROTTEN_PREDICTION", "high_confidence_rotten_prediction"],
  "manual_review_required": false,
  "top_predictions": [
    {
      "class_name": "Banana__Rotten",
      "confidence": 0.91
    },
    {
      "class_name": "Mango__Rotten",
      "confidence": 0.05
    }
  ],
  "prediction": {
    "predicted_class": "Banana__Rotten",
    "product_type": "Banana",
    "condition": "rotten",
    "confidence": 0.91,
    "top_k": [],
    "top1_top2_margin": 0.86
  },
  "quality": {
    "grade": "C",
    "overall_quality_score": 18.72,
    "component_scores": {
      "model_condition": 9.0,
      "color": 72.1,
      "defect_absence": 9.0,
      "image_quality": 84.7,
      "size_proxy": 75.0,
      "ripeness": 31.1
    },
    "action": "manual_review_or_discard",
    "inventory_status": "blocked_pending_review",
    "discount_percentage": null,
    "manual_review": false,
    "reason_codes": ["classified_as_rotten", "high_confidence_rotten_prediction"],
    "warnings": [
      "size_proxy_is_relative_to_image_area_not_physical_size",
      "colour_score_is_lighting_sensitive_proxy",
      "quality_grade_is_rule_based_not_supervised_model_output"
    ]
  },
  "xai": {
    "method": "Grad-CAM",
    "available": false,
    "heatmap_path": null,
    "note": "Grad-CAM is report-facing attention evidence and is not used for quality grading."
  },
  "model_info": {
    "model_name": "efficientnetb0_aug_oversampled_finetuned_wsl",
    "model_version": "2026-04-29T15:45:22"
  }
}
```

Runtime safeguards:

- classifier confidence below `0.60` maps the quality decision to `Review`
- top-1/top-2 margin below `0.15` triggers manual review
- same-produce healthy/rotten conflict triggers manual review
- high-confidence rotten predictions map to Grade `C`

The runtime API cannot know whether a prediction is wrong. High-confidence
error labels are only valid in evaluation notebooks where ground truth exists.
The quality grade is rule-based and must not be described as a supervised
EfficientNetB0 grade prediction.

### POST /feedback

Logs user corrections or overrides for monitoring and future retraining.

Example request:

```json
{
  "prediction_id": "uuid",
  "image_id": "optional-client-image-id",
  "predicted_class": "Banana__Rotten",
  "predicted_grade": "C",
  "corrected_class": "Banana__Healthy",
  "producer_override_class": "Banana__Healthy",
  "producer_override_grade": "Review",
  "override_reason": "Producer requested manual inspection after visual check.",
  "accepted_ai_recommendation": false,
  "quality_decision_snapshot": {
    "overall_quality_score": 18.72,
    "component_scores": {
      "model_condition": 9.0
    },
    "reason_codes": ["high_confidence_rotten_prediction"]
  },
  "model_name": "efficientnetb0_aug_oversampled_finetuned_wsl",
  "user_note": "Producer corrected the freshness status after inspection."
}
```

Example response:

```json
{
  "status": "logged"
}
```

### GET /monitoring/feedback-summary

Returns a post-deployment monitoring summary built from `predictions.jsonl` and
`feedback.jsonl`. The service joins records by `prediction_id` and calculates a
human-feedback accuracy proxy over time.

Example request:

```text
GET /monitoring/feedback-summary
```

Example response:

```json
{
  "monitoring_type": "human_feedback_accuracy_proxy",
  "prediction_log_count": 5,
  "feedback_log_count": 1,
  "labelled_class_feedback_count": 1,
  "class_accuracy_proxy": 1.0,
  "labelled_grade_feedback_count": 1,
  "grade_accuracy_proxy": 1.0,
  "accepted_feedback_count": 1,
  "override_feedback_count": 0,
  "override_rate": 0.0,
  "high_confidence_override_count": 0,
  "daily": [
    {
      "date": "2026-05-01",
      "feedback_count": 1,
      "class_accuracy_proxy": 1.0,
      "grade_accuracy_proxy": 1.0,
      "override_rate": 0.0
    }
  ],
  "limitations": [
    "feedback_is_sparse_and_may_be_biased",
    "accepted_or_overridden_feedback_is_not_a_controlled_ground_truth_test_set",
    "automatic_retraining_is_not_performed"
  ]
}
```

Interpretation:

- `class_accuracy_proxy` is based on accepted/corrected class feedback.
- `grade_accuracy_proxy` is based on accepted/corrected grade feedback.
- `override_rate` tracks how often humans reject or change the AI decision.
- `high_confidence_override_count` flags potentially dangerous confident mistakes.

This endpoint is the Task 3 monitoring interface. It should not be described as
controlled test-set accuracy because feedback is sparse, selective and may be
noisy. Its purpose is to monitor accuracy trends, drift risk, weak classes and
high-confidence mistakes over time.

### GET /recommend/reorder

Returns Task 1 quick-reorder recommendations for a DESD customer ID.

Example request:

```text
GET /recommend/reorder?customer_id=C000003&top_k=3
```

Optional query parameter:

```text
method=global_popularity|user_frequency|frequency_recency
```

Set `include_discovery=true` to include the optional "You may also like" branch:

```text
GET /recommend/reorder?customer_id=C000003&top_k=3&include_discovery=true
```

Example response:

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
      "score": 0.801946,
      "reason_codes": [
        "frequently_ordered_by_customer",
        "ordered_recently"
      ],
      "reason_text": "Recommended because it is ordered repeatedly by this customer, ordered recently by this customer."
    }
  ],
  "limitations": [
    "recommendations_based_on_synthetic_seed_data",
    "not_production_customer_behaviour"
  ]
}
```

Hybrid response includes:

```json
{
  "recommendations": [],
  "quick_reorder": [],
  "you_may_also_like": [
    {
      "rank": 1,
      "product_id": "P000096",
      "product_name": "New Season Potatoes",
      "producer_id": "PR000003",
      "score": 0.973333,
      "score_components": {
        "cooccurrence": 0.82,
        "seasonality": 0.08,
        "segment_popularity": 0.073333,
        "diversity_adjustment": 0.0
      },
      "based_on_product_ids": ["P000106", "P000119", "P000097"],
      "reason_codes": [
        "commonly_bought_together",
        "similar_basket_pattern",
        "new_to_customer"
      ]
    }
  ]
}
```

This endpoint is powered by fake/synthetic DESD seed data. It is suitable for
demo integration and proof-of-concept evaluation, not production customer
behaviour claims.

## DESD Integration Pattern

DESD should:

1. receive an image upload from a producer or admin;
2. send the image to `POST /predict`;
3. display prediction, confidence, grade, action, reason codes, and manual-review flag;
4. store the returned `prediction_id` with the DESD record;
5. call `POST /feedback` if a human overrides the prediction.
6. call `GET /monitoring/feedback-summary` for admin monitoring of feedback-based accuracy proxies.
7. call `GET /recommend/reorder` to display quick-reorder suggestions for customers.

The model must be presented as decision support, not autonomous quality
approval.
