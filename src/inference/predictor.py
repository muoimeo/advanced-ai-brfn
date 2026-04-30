from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import CLASS_NAMES_PATH, IMAGE_SIZE, MODELS_DIR, MODEL_METADATA_PATH
from src.inference.postprocess import build_business_prediction
from src.models.resnet50_model import ResNet50Preprocess
from src.quality.image_features import extract_image_features
from src.quality.rules import (
    build_quality_decision,
    image_features_to_dict,
    model_prediction_to_dict,
    quality_decision_to_dict,
    top_k_from_dicts,
)
from src.quality.schemas import ModelPrediction


BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"


def load_class_names(path: Path = CLASS_NAMES_PATH) -> list[str]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_model_metadata(path: Path = MODEL_METADATA_PATH) -> dict:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_model() -> tf.keras.Model:
    custom_objects = {"ResNet50Preprocess": ResNet50Preprocess}
    return tf.keras.models.load_model(
        BEST_MODEL_PATH,
        custom_objects=custom_objects,
        compile=False,
    )


@lru_cache(maxsize=1)
def get_class_names() -> list[str]:
    return load_class_names()


@lru_cache(maxsize=1)
def get_metadata() -> dict:
    return load_model_metadata()


def image_bytes_to_tensor(image_bytes: bytes) -> tf.Tensor:
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)


def predict_image_bytes(image_bytes: bytes, top_k: int = 3) -> dict:
    model = get_model()
    class_names = get_class_names()
    metadata = get_metadata()
    image_tensor = image_bytes_to_tensor(image_bytes)

    probabilities = model.predict(image_tensor, verbose=0)[0]
    top_k = max(1, min(top_k, len(class_names)))
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    predicted_index = int(top_indices[0])
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])
    top_predictions = [
        {
            "class_name": class_names[int(index)],
            "confidence": float(probabilities[int(index)]),
        }
        for index in top_indices
    ]
    top1_top2_margin = None
    if len(top_predictions) >= 2:
        top1_top2_margin = (
            top_predictions[0]["confidence"] - top_predictions[1]["confidence"]
        )

    prediction = build_business_prediction(
        predicted_class=predicted_class,
        confidence=confidence,
        top_predictions=top_predictions,
        top1_top2_margin=top1_top2_margin,
    )
    prediction["model_name"] = metadata.get("model_name")
    prediction["model_version"] = metadata.get("trained_at") or metadata.get("selected_at")
    model_prediction = ModelPrediction(
        predicted_class=predicted_class,
        product_type=prediction["produce_type"],
        condition=prediction["freshness_status"],
        confidence=confidence,
        top_k=top_k_from_dicts(top_predictions),
        top1_top2_margin=top1_top2_margin,
    )
    image_features = extract_image_features(
        image_bytes=image_bytes,
        product_type=prediction["produce_type"],
    )
    quality_decision = build_quality_decision(model_prediction, image_features)
    quality = quality_decision_to_dict(quality_decision)
    prediction_object = model_prediction_to_dict(model_prediction)

    prediction["quality"] = quality
    prediction["prediction"] = prediction_object
    prediction["image_features"] = image_features_to_dict(image_features)
    prediction["xai"] = {
        "method": "Grad-CAM",
        "available": False,
        "heatmap_path": None,
        "note": "Grad-CAM is report-facing attention evidence and is not used for quality grading.",
    }
    prediction["model_info"] = {
        "model_name": prediction["model_name"],
        "model_version": prediction["model_version"],
    }

    # Legacy flat fields are retained for current demo/tests while quality becomes authoritative.
    prediction["quality_grade"] = quality["grade"]
    prediction["recommended_action"] = quality["action"]
    prediction["manual_review_required"] = (
        prediction["manual_review_required"] or quality["manual_review"]
    )
    prediction["reason_codes"] = list(
        dict.fromkeys([*prediction["reason_codes"], *quality["reason_codes"]])
    )
    return prediction
