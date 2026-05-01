# BRFN AI Service Demo Script

This demo shows the Advanced AI component as a standalone FastAPI service that
DESD can call over HTTP. The model is frozen as EfficientNetB0 fine-tuned; the
quality grade is produced by the external rule-based layer in `src/quality/`.

## Pre-demo Checks

Run from the repository root:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

In another terminal:

```bash
pytest
```

Expected status: all tests pass. The current warnings are pandas/numpy
deprecation warnings, not application failures.

## Demo Images

The demo requests use local images from:

```text
data/raw/custom_test_images/
```

Recommended examples:

- `hea-app.jpg`: healthy apple, expected Grade A with image-quality review warning.
- `mid_banana.jpg`: healthy banana, expected Grade B because colour/size proxy do
  not meet all Grade A thresholds.
- `mid_orange.jpg`: healthy orange, expected Grade B because size proxy is below
  the Grade A threshold.
- `rot-ban1.jpg`: rotten banana, expected Grade C and blocked/manual handling.

## Flow

1. Call `GET /health`.
   - Confirms the API process is running and whether the model can load.

2. Call `GET /model-info`.
   - Shows the final selected model, metrics, grouped split strategy, XAI risk
     audit, and quality-layer metadata.

3. Call `POST /predict` with `hea-app.jpg`.
   - Explain the nested response:
     - `prediction`: EfficientNetB0 class evidence.
     - `quality`: rule-based grade/action decision.
     - `xai`: reference only; Grad-CAM is not used for grading.

4. Call `POST /predict` with `mid_banana.jpg` or `mid_orange.jpg`.
   - Show a Grade B case.
   - Explain that Grade B means acceptable but not Grade A, so the action is
     `discount_or_quick_sale`.

5. Call `POST /predict` with `rot-ban1.jpg`.
   - Show conservative handling for rotten produce: Grade C, blocked pending
     review/discard decision.

6. Call `POST /feedback`.
   - Demonstrates accountability. DESD can store the `prediction_id` and submit
     a producer/admin override without storing unnecessary personal data.

## Talking Points

- The CNN does not predict Grade A/B/C. It predicts the 28-class
  produce/freshness label.
- The quality layer combines model evidence with image proxy features:
  colour score, dark ratio, blur/brightness, foreground coverage, size proxy.
- Grad-CAM is used for model audit and report evidence, not runtime grading.
- The system is suitable as decision support, not autonomous quality approval.
- Manual-review flags are deliberate deployment safeguards for domain shift,
  poor image quality, low confidence, or ambiguous predictions.
