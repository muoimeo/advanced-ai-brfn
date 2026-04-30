# Advanced AI Produce Freshness Classifier

This repository contains the Advanced AI coursework project for fruit and vegetable freshness / rotten image classification.

The project uses the Kaggle dataset:

```text
muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
```

The current framing is a 28-class image classification task. Downstream inference maps the raw class prediction into business-friendly fields such as produce type, freshness status, quality grade, and recommended action.

Quality grading is implemented as a separate rule-based layer outside the CNN.
EfficientNetB0 predicts product type and healthy/rotten condition; `src/quality/`
combines that model evidence with transparent image proxy features such as
colour profile, dark-area ratio, blur, brightness, and foreground coverage.
The project must not claim the model learned supervised Grade A/B/C labels.

## Current Status

The final selected deployment candidate is:

```text
efficientnetb0_aug_oversampled_finetuned_wsl
```

Final comparison uses `data/splits_grouped` to avoid leakage from
offline-augmented variants of the same source image crossing train,
validation, and test sets. ResNet50 fine-tuned remains the main macro-F1
challenger, but EfficientNetB0 is selected for the final demo because it has a
better deployment-risk tradeoff in the grouped XAI audit.

## Local Workflow

This project now trains and evaluates locally.

1. Create and activate a virtual environment.
2. Install dependencies.
3. Use `data/splits_grouped/` for final comparison and reporting.
4. Keep the selected model in `models/best_model.keras`.
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

Do not commit the full raw dataset.

## Expected Artifacts

Generated locally:

```text
data/splits_grouped/train.csv
data/splits_grouped/val.csv
data/splits_grouped/test.csv
models/class_names.json
models/best_model.keras
models/model_metadata.json
outputs/eda/
outputs/figures/
outputs/confusion_matrices/
outputs/xai_examples/
outputs/quality_rule_eval/
docs/report_figures/custom_image_validation.csv
docs/report_figures/custom_image_validation_summary.csv
```

## Experiment Scope

Preferred comparison set:

1. baseline custom CNN
2. MobileNetV2
3. ResNet50
4. EfficientNetB0

## API Goal

The service exposes:

- `GET /health`
- `POST /predict`
- `POST /feedback`
- `GET /model-info`

The DESD system can later call this API for image inference. See
`docs/api_contract.md` for the response schema and integration pattern.

The preferred response contract is nested:

- `prediction`: model class evidence
- `quality`: rule-based grade/action decision
- `xai`: optional attention reference, not used for grading
- `model_info`: selected model metadata
