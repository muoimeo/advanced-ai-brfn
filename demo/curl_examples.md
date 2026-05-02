# Curl Demo Examples

This file provides tested local API demo commands for the BRFN Advanced AI service.

The examples below are written for **Windows PowerShell**. In PowerShell, `curl` is often an alias for `Invoke-WebRequest`, so use `curl.exe` explicitly.

---

## 1. Start the API

Run this from the project root:

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

Expected behaviour:

* API starts on `http://localhost:8001`
* Model metadata is loaded
* `/health` and `/model-info` become available

---

## 2. Health and model metadata

```powershell
curl.exe http://localhost:8001/health
```

```powershell
curl.exe http://localhost:8001/model-info
```

Optional pretty JSON output:

```powershell
curl.exe -s http://localhost:8001/model-info | python -m json.tool
```

---

## 3. Prediction examples

Make sure the sample image files exist before running prediction requests:

```powershell
Test-Path "data\raw\custom_test_images\hea-app.jpg"
Test-Path "data\raw\custom_test_images\mid_banana.jpg"
Test-Path "data\raw\custom_test_images\rot-ban1.jpg"
```

Each command should return `True`.

### 3.1 Healthy apple example

```powershell
curl.exe -v "http://localhost:8001/predict" `
  -F "file=@data/raw/custom_test_images/hea-app.jpg;type=image/jpeg"
```

Pretty JSON version:

```powershell
curl.exe -s "http://localhost:8001/predict" `
  -F "file=@data/raw/custom_test_images/hea-app.jpg;type=image/jpeg" `
  | python -m json.tool
```

### 3.2 Mid-quality banana example

```powershell
curl.exe -v "http://localhost:8001/predict" `
  -F "file=@data/raw/custom_test_images/mid_banana.jpg;type=image/jpeg"
```

Pretty JSON version:

```powershell
curl.exe -s "http://localhost:8001/predict" `
  -F "file=@data/raw/custom_test_images/mid_banana.jpg;type=image/jpeg" `
  | python -m json.tool
```

### 3.3 Rotten banana example

```powershell
curl.exe -v "http://localhost:8001/predict" `
  -F "file=@data/raw/custom_test_images/rot-ban1.jpg;type=image/jpeg"
```

Pretty JSON version:

```powershell
curl.exe -s "http://localhost:8001/predict" `
  -F "file=@data/raw/custom_test_images/rot-ban1.jpg;type=image/jpeg" `
  | python -m json.tool
```

### 3.4 Save prediction outputs for demo evidence

```powershell
New-Item -ItemType Directory -Force -Path "demo\outputs"
```

```powershell
curl.exe -s "http://localhost:8001/predict" `
  -F "file=@data/raw/custom_test_images/hea-app.jpg;type=image/jpeg" `
  | python -m json.tool > "demo/outputs/hea-app_prediction.json"
```

```powershell
curl.exe -s "http://localhost:8001/predict" `
  -F "file=@data/raw/custom_test_images/mid_banana.jpg;type=image/jpeg" `
  | python -m json.tool > "demo/outputs/mid_banana_prediction.json"
```

```powershell
curl.exe -s "http://localhost:8001/predict" `
  -F "file=@data/raw/custom_test_images/rot-ban1.jpg;type=image/jpeg" `
  | python -m json.tool > "demo/outputs/rot-ban1_prediction.json"
```

Check saved files:

```powershell
Get-ChildItem "demo\outputs"
```

---

## 4. Expected prediction response structure

A successful `/predict` response should contain these main sections:

```json
{
  "prediction_id": "uuid",
  "prediction": {
    "predicted_class": "Banana__Healthy",
    "product_type": "Banana",
    "condition": "healthy",
    "confidence": 0.987,
    "top_k": [],
    "top1_top2_margin": 0.95
  },
  "quality": {
    "grade": "B",
    "overall_quality_score": 85.77,
    "component_scores": {
      "model_condition": 98.73,
      "color": 83.01,
      "defect_absence": 100.0,
      "image_quality": 18.43,
      "size_proxy": 75.0,
      "ripeness": 93.42
    },
    "action": "discount_or_quick_sale",
    "inventory_status": "surplus_or_lower_grade",
    "discount_percentage": 20,
    "manual_review": true,
    "reason_codes": [],
    "warnings": []
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

Notes:

* `prediction` is the EfficientNetB0 classification output.
* `quality` is the external rule-based grading layer output.
* `xai` is metadata only in API runtime. Grad-CAM is not used to calculate grade.
* `quality.grade` is rule-based. It is not a supervised Grade A/B/C model output.

---

## 5. Feedback example

The `/feedback` endpoint stores producer/admin acceptance, correction, or override evidence for monitoring and future retraining. It does **not** retrain the model automatically.

PowerShell can break multi-line JSON when using `-d '{...}'` directly. The safest workflow is:

1. Create a JSON payload file.
2. Validate the JSON.
3. Send the file using `--data-binary`.
4. Check the feedback log.

### 5.1 Create feedback payload file

Replace `prediction_id` with an ID returned by `/predict`.

```powershell
@'
{
  "prediction_id": "replace-with-prediction-id",
  "predicted_class": "Banana__Healthy",
  "predicted_grade": "B",
  "producer_override_class": "Banana__Healthy",
  "producer_override_grade": "B",
  "override_reason": "Producer reviewed the image and accepted the AI Grade B recommendation.",
  "accepted_ai_recommendation": true,
  "quality_decision_snapshot": {
    "overall_quality_score": 85.77,
    "component_scores": {
      "model_condition": 98.73,
      "color": 83.01,
      "defect_absence": 100.0,
      "image_quality": 18.43,
      "size_proxy": 75.0,
      "ripeness": 93.42
    },
    "reason_codes": [
      "classified_as_healthy",
      "acceptable_quality_but_not_grade_a",
      "image_quality_warning",
      "manual_review_recommended"
    ],
    "warnings": [
      "size_proxy_is_relative_to_image_area_not_physical_size",
      "colour_score_is_lighting_sensitive_proxy",
      "quality_grade_is_rule_based_not_supervised_model_output"
    ],
    "action": "discount_or_quick_sale"
  }
}
'@ | Set-Content -Path "demo\feedback_payload.json" -Encoding utf8
```

### 5.2 Validate JSON

```powershell
Get-Content demo\feedback_payload.json | python -m json.tool
```

If this command prints formatted JSON, the payload is valid.

### 5.3 Send feedback

```powershell
curl.exe -i "http://localhost:8001/feedback" `
  -H "Content-Type: application/json" `
  --data-binary "@demo/feedback_payload.json"
```

Successful response:

```json
{"status":"logged"}
```

### 5.4 Check feedback log

```powershell
Get-Content outputs/logs/feedback.jsonl -Tail 3
```

The latest log line should include fields such as:

```json
{
  "prediction_id": "replace-with-prediction-id",
  "predicted_class": "Banana__Healthy",
  "predicted_grade": "B",
  "producer_override_class": "Banana__Healthy",
  "producer_override_grade": "B",
  "accepted_ai_recommendation": true,
  "quality_decision_snapshot": {
    "overall_quality_score": 85.77,
    "component_scores": {},
    "reason_codes": [],
    "warnings": [],
    "action": "discount_or_quick_sale"
  }
}
```

---

## 6. Optional feedback override example

Use this example when the producer accepts the predicted class but overrides the grade/action because the image-quality warning suggests manual review.

```powershell
@'
{
  "prediction_id": "replace-with-prediction-id",
  "predicted_class": "Banana__Healthy",
  "predicted_grade": "B",
  "producer_override_class": "Banana__Healthy",
  "producer_override_grade": "Review",
  "override_reason": "Producer accepted the produce class but requested manual review after the image-quality warning.",
  "accepted_ai_recommendation": false,
  "quality_decision_snapshot": {
    "overall_quality_score": 85.77,
    "component_scores": {
      "model_condition": 98.73,
      "color": 83.01,
      "defect_absence": 100.0,
      "image_quality": 18.43,
      "size_proxy": 75.0,
      "ripeness": 93.42
    },
    "reason_codes": [
      "classified_as_healthy",
      "acceptable_quality_but_not_grade_a",
      "image_quality_warning",
      "manual_review_recommended"
    ],
    "warnings": [
      "size_proxy_is_relative_to_image_area_not_physical_size",
      "colour_score_is_lighting_sensitive_proxy",
      "quality_grade_is_rule_based_not_supervised_model_output"
    ],
    "action": "discount_or_quick_sale"
  }
}
'@ | Set-Content -Path "demo_feedback_override_payload.json" -Encoding utf8
```

```powershell
Get-Content demo_feedback_override_payload.json | python -m json.tool
```

```powershell
curl.exe -i "http://localhost:8001/feedback" `
  -H "Content-Type: application/json" `
  --data-binary "@demo_feedback_override_payload.json"
```

---

## 7. Troubleshooting

## 7. Quick reorder recommendation example

The `/recommend/reorder` endpoint is for Task 1. It uses fake/synthetic DESD seed export data and returns quick-reorder suggestions with reason codes.

Use an existing customer ID from `data\task1\desd_seed_export\customers.csv`, for example:

```powershell
curl.exe -s "http://localhost:8001/recommend/reorder?customer_id=C000003&top_k=3" `
  | python -m json.tool
```

Hybrid quick reorder + "You may also like" discovery:

```powershell
curl.exe -s "http://localhost:8001/recommend/reorder?customer_id=C000003&top_k=3&include_discovery=true" `
  | python -m json.tool
```

Optional method comparison:

```powershell
curl.exe -s "http://localhost:8001/recommend/reorder?customer_id=C000003&top_k=3&method=global_popularity" `
  | python -m json.tool
```

```powershell
curl.exe -s "http://localhost:8001/recommend/reorder?customer_id=C000003&top_k=3&method=user_frequency" `
  | python -m json.tool
```

```powershell
curl.exe -s "http://localhost:8001/recommend/reorder?customer_id=C000003&top_k=3&method=frequency_recency" `
  | python -m json.tool
```

Expected response fields:

```json
{
  "customer_id": "C000003",
  "recommendation_date": "2026-05-02",
  "method": "frequency_recency",
  "top_k": 3,
  "recommendations": [
    {
      "product_id": "P000113",
      "product_name": "Autumn Raspberries",
      "producer_id": "PR000008",
      "score": 0.801946,
      "reason_codes": ["frequently_ordered_by_customer", "ordered_recently"]
    }
  ],
  "limitations": [
    "recommendations_based_on_synthetic_seed_data",
    "not_production_customer_behaviour"
  ]
}
```

When `include_discovery=true`, the response also contains:

```json
{
  "quick_reorder": [],
  "you_may_also_like": [
    {
      "product_id": "P000096",
      "product_name": "New Season Potatoes",
      "producer_id": "PR000003",
      "score_components": {
        "cooccurrence": 0.82,
        "seasonality": 0.08,
        "segment_popularity": 0.073333,
        "diversity_adjustment": 0.0
      },
      "based_on_product_ids": ["P000106", "P000119", "P000097"],
      "reason_codes": ["commonly_bought_together", "new_to_customer"]
    }
  ]
}
```

---

## 8. Troubleshooting

### PowerShell says: `A parameter cannot be found that matches parameter name 'X'`

You used `curl` instead of `curl.exe`. Use:

```powershell
curl.exe http://localhost:8001/health
```

### Curl says: `Could not resolve host: reviewed`

PowerShell split a multi-line JSON body into separate arguments. Use the payload-file approach with `--data-binary "@file.json"`.

### Curl says: `Failed to open/read local data from file/application`

The image path is wrong. Check file existence:

```powershell
Test-Path "data\raw\custom_test_images\hea-app.jpg"
```

### API returns `422 Unprocessable Entity`

The request body does not match the endpoint schema. For `/predict`, ensure the upload field is named `file`:

```powershell
-F "file=@data/raw/custom_test_images/hea-app.jpg;type=image/jpeg"
```

For `/feedback`, validate the payload file first:

```powershell
Get-Content demo\feedback_payload.json | python -m json.tool
```

### API returns `500 Internal Server Error`

Check the terminal running `uvicorn`. Common causes:

* missing model file
* wrong model metadata path
* missing `quality_profiles.yaml`
* image read error
* schema mismatch between predictor output and quality layer

---

## 9. Demo checklist

Before the live demo, verify:

```powershell
curl.exe http://localhost:8001/health
curl.exe http://localhost:8001/model-info
```

Then run at least one successful request for each endpoint:

```text
GET  /health
GET  /model-info
POST /predict
POST /feedback
GET  /recommend/reorder
```

Recommended demo cases:

```text
hea-app.jpg      -> healthy produce example
mid_banana.jpg   -> Grade B / discount-or-quick-sale example
rot-ban1.jpg     -> rotten / Grade C or manual-review example
```

Evidence to capture:

```text
- /health response
- /model-info response
- /predict response for Grade A/B/C or Review cases
- /feedback response {"status":"logged"}
- /recommend/reorder response with product, producer, score, and reason codes
- outputs/logs/feedback.jsonl showing logged feedback
- pytest result showing all tests passed
```
