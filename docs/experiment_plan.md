# Experiment Plan

## Objective

Build and compare image classifiers for the 28-class fruit and vegetable healthy/rotten dataset.

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

The raw dataset must not be committed to Git.

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

Default split:

- train: 70%
- validation: 15%
- test: 15%

The split is stratified by `class_name`.

## Model Comparison

Models:

1. baseline custom CNN
2. MobileNetV2
3. ResNet50
4. EfficientNetB0

Rationale:

- the baseline CNN provides a simple from-scratch reference
- transfer learning models provide stronger candidates for limited coursework time
- the comparison remains broad enough without becoming unmanageable

## Metrics

Primary metrics:

- accuracy
- macro F1
- precision
- recall

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

`model_metadata.json` should include:

- model name
- dataset slug
- number of classes
- image size
- split ratios
- metrics
- training date
- notes on limitations

## Next Implementation Steps

1. Run local EDA notebook.
2. Confirm the dataset has 28 classes.
3. Implement or complete local baseline training notebook.
4. Save baseline metrics and confusion matrix.
5. Repeat for transfer learning models.
6. Select the best model based on test metrics and practical inference behavior.
7. Build inference API around the final exported model.
