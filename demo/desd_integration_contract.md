# DESD Integration Contract

The DESD repository should call this AI service over HTTP. Do not merge the AI
model code into DESD.

## Service URL

Local demo:

```text
http://localhost:8001
```

Docker Compose network:

```text
http://ai-service:8001
```

## Prediction Request

DESD uploads one image using `multipart/form-data`:

```http
POST /predict
Content-Type: multipart/form-data

file=<uploaded-image>
```

DESD should store:

- `prediction_id`
- `prediction.predicted_class`
- `prediction.product_type`
- `prediction.condition`
- `prediction.confidence`
- `quality.grade`
- `quality.action`
- `quality.inventory_status`
- `quality.discount_percentage`
- `quality.manual_review`
- `quality.reason_codes`
- `quality.warnings`
- `model_info.model_name`
- `model_info.model_version`

## Feedback Request

If a producer/admin overrides the AI recommendation, DESD sends:

```http
POST /feedback
Content-Type: application/json
```

Minimum useful body:

```json
{
  "prediction_id": "uuid-from-predict",
  "predicted_class": "Banana__Healthy",
  "predicted_grade": "B",
  "producer_override_class": "Banana__Healthy",
  "producer_override_grade": "Review",
  "override_reason": "Manual inspection requested.",
  "accepted_ai_recommendation": false,
  "quality_decision_snapshot": {
    "overall_quality_score": 85.77,
    "component_scores": {
      "model_condition": 98.73,
      "color": 83.01,
      "image_quality": 18.43
    },
    "reason_codes": ["acceptable_quality_but_not_grade_a"],
    "warnings": ["quality_grade_is_rule_based_not_supervised_model_output"],
    "action": "discount_or_quick_sale"
  },
  "model_name": "efficientnetb0_aug_oversampled_finetuned_wsl"
}
```

## UI Guidance

DESD should present the result as decision support:

- show class, confidence, grade, action, and manual-review flag;
- show reason codes and warnings in admin/producer-facing detail;
- avoid wording that implies autonomous approval;
- require manual confirmation for Grade `C` and `Review`.
