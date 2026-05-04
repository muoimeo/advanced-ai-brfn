# Final Submission Plan

## Objective

Prepare the BRFN freshness classifier for final Advanced AI submission and
demo. The training comparison is complete; the current focus is evidence
packaging, custom-image validation, inference/API hardening, and deployment
readiness.

The model output remains a 28-class prediction. Business logic can later aggregate the output into produce type, freshness status, quality grade, and recommended action.

## Environment

Experiments run locally in this repository.

Local responsibilities:

- dataset download or local dataset path configuration
- EDA
- split generation
- training
- evaluation
- artifact export
- inference API testing

The raw dataset is excluded from Git.

## Dataset

Source:

```text
muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
```

Expected structure after locating the dataset root:

```text
dataset_root/
  class_1/
    image.jpg
  class_2/
    image.jpg
  ...
```

Expected number of classes: 28.

## EDA Steps

Notebook: `notebooks/01_eda.ipynb`

1. Locate or download the dataset locally.
2. Inspect folder structure.
3. Find the dataset root containing class folders.
4. Build an image dataframe.
5. Verify 28 classes.
6. Count images per class.
7. Plot class distribution.
8. Show sample images.
9. Create stratified train/validation/test splits.
10. Save split CSVs and class names.

Expected outputs:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
models/class_names.json
outputs/eda/class_counts.csv
outputs/eda/split_counts.csv
outputs/eda/eda_metadata.json
```

## Split Strategy

Current selected split:

- directory: `data/splits_grouped`
- strategy: grouped source-image split
- group key: `source_image_id`
- purpose: prevent offline-augmented variants of the same source image from crossing train/validation/test

The older `data/splits` artifacts are retained for traceability, while final
top-model comparison and reporting use the grouped split.

## Model Comparison

Completed model families:

1. baseline custom CNN
2. MobileNetV2
3. EfficientNetB0
4. ResNet50

Rationale:

- the baseline CNN provides a simple from-scratch reference
- transfer learning models provide stronger candidates for limited coursework time
- the comparison remains broad enough without becoming unmanageable
- final selection considers XAI and deployment risk, not accuracy alone

Current selected final model:

```text
efficientnetb0_aug_oversampled_finetuned_wsl
```

ResNet50 fine-tuned remains the main challenger because it has stronger macro
F1. EfficientNetB0 fine-tuned remains selected because it has a better
deployment-risk profile in the grouped XAI audit, including fewer
high-confidence errors.

## Metrics

Primary metrics:

- accuracy
- macro F1
- precision
- recall
- per-class F1
- high-confidence error count
- custom-image produce correctness
- custom-image freshness correctness
- custom-image full 28-class correctness

Supporting outputs:

- confusion matrix
- classification report
- train/validation curves
- example predictions

Macro F1 is important because class imbalance may affect plain accuracy.

## Artifact Export

Final selected model:

```text
models/best_model.keras
models/class_names.json
models/model_metadata.json
```

`model_metadata.json` includes:

- model name
- dataset slug
- number of classes
- image size
- split ratios
- metrics
- training date
- notes on limitations

## Current Implementation Steps

1. Freeze the EfficientNetB0 final-model decision unless a clear bug is found.
2. Convert `custom_image_test.ipynb` into a report-ready validation table using `docs/report_figures/custom_image_manifest.csv`.
3. Add inference decision rules for low confidence and small top-1/top-2 margin.
4. Ensure `/predict` returns model metadata, business mapping, top predictions, reason codes, and manual-review flags.
5. Extend deterministic tests for post-processing, prediction schema, API validation, and logging.
6. Update capture guidance and report notes around domain shift, background bias, object scale mismatch, and manual-review mitigation.
7. Dockerise the FastAPI service for CPU inference after API behavior is stable.
8. Document the DESD integration contract as an HTTP API call, without merging repositories.
