# Advanced AI Produce Freshness Classifier

This repository contains the Advanced AI coursework project for fruit and vegetable freshness / rotten image classification.

The project uses the Kaggle dataset:

```text
muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
```

The current framing is a 28-class image classification task. Downstream inference maps the raw class prediction into business-friendly fields such as produce type, freshness status, quality grade, and recommended action.

## Local Workflow

This project now trains and evaluates locally.

1. Create and activate a virtual environment.
2. Install dependencies.
3. Run `notebooks/01_eda.ipynb` locally.
4. Generate local split CSV files in `data/splits/`.
5. Train baseline and transfer learning notebooks locally.
6. Save the selected model to `models/best_model.keras`.
7. Save class names and metadata to `models/`.
8. Run the local inference API.

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
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
models/class_names.json
models/best_model.keras
models/model_metadata.json
outputs/eda/
outputs/figures/
outputs/confusion_matrices/
```

## Experiment Scope

Preferred comparison set:

1. baseline custom CNN
2. MobileNetV2
3. ResNet50
4. EfficientNetB0

## API Goal

The final service should expose:

- `GET /health`
- `POST /predict`
- `POST /feedback`
- `GET /model-info`

The DESD system can later call this API for image inference.
