from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMAGE_SIZE
from src.data.dataloaders import normalize_image_paths_for_runtime
from src.inference.postprocess import split_class_name


def load_image_for_model(
    image_path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[tf.Tensor, np.ndarray, str]:
    """Load one image as a 0..1 batch tensor and display-ready RGB array."""
    runtime_path = normalize_image_paths_for_runtime([str(image_path)])[0]
    image_bytes = tf.io.read_file(runtime_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0

    image_array = image.numpy()
    image_batch = tf.expand_dims(image, axis=0)
    return image_batch, image_array, runtime_path


def find_nested_base_model(
    model: keras.Model,
    name_prefix: str = "resnet50",
) -> keras.Model:
    candidates = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model) and layer.name.startswith(name_prefix)
    ]

    if not candidates:
        raise ValueError(f"Could not find nested base model starting with {name_prefix!r}.")

    return candidates[0]


def find_last_conv_layer(
    model: keras.Model,
    name_prefix: str = "resnet50",
) -> tuple[keras.Model, str]:
    base_model = find_nested_base_model(model, name_prefix=name_prefix)

    for layer in reversed(base_model.layers):
        if isinstance(layer, layers.Conv2D):
            return base_model, layer.name

    raise ValueError(f"Could not find a Conv2D layer inside {base_model.name}.")


def select_gradcam_layer(
    model: keras.Model,
    name_prefix: str,
    preferred_layer_names: list[str] | None = None,
) -> tuple[keras.Model, str]:
    """Return the first preferred layer that exists, otherwise the last Conv2D layer."""
    base_model = find_nested_base_model(model, name_prefix=name_prefix)
    existing_layer_names = {layer.name for layer in base_model.layers}

    for layer_name in preferred_layer_names or []:
        if layer_name in existing_layer_names:
            return base_model, layer_name

    return find_last_conv_layer(model, name_prefix=name_prefix)


def format_class_name_for_business(class_name: str) -> str:
    produce_type, freshness_status = split_class_name(class_name)
    produce_text = produce_type.replace("_", " ").strip().lower()

    if freshness_status == "healthy":
        return f"healthy {produce_text}"
    if freshness_status == "rotten":
        return f"rotten {produce_text}"
    return class_name.replace("__", " ").replace("_", " ").strip().lower()


def build_xai_caption(
    predicted_class: str,
    confidence: float,
    true_class: str | None = None,
    correct: bool | None = None,
) -> str:
    prediction = format_class_name_for_business(predicted_class)
    lines = [
        f"Model prediction: {prediction}",
        f"Confidence: {confidence:.1%}",
    ]

    if true_class is not None:
        truth = format_class_name_for_business(true_class)
        lines.insert(0, f"Actual label: {truth}")

    if correct is False and confidence >= 0.95:
        lines.append("Risk: high-confidence error, manual review needed.")
    elif correct is False:
        lines.append("Risk: incorrect prediction, inspect the attention map.")
    else:
        lines.append("Interpretation: highlighted regions influenced the decision most.")

    return "\n".join(lines)


def _call_layer(layer: keras.layers.Layer, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    try:
        return layer(inputs, training=training)
    except TypeError:
        return layer(inputs)


def _split_model_around_base(
    model: keras.Model,
    base_model: keras.Model,
) -> tuple[list[keras.layers.Layer], list[keras.layers.Layer]]:
    base_index = model.layers.index(base_model)
    prefix_layers = [
        layer
        for layer in model.layers[:base_index]
        if not isinstance(layer, layers.InputLayer)
    ]
    suffix_layers = [
        layer
        for layer in model.layers[base_index + 1 :]
        if not isinstance(layer, layers.InputLayer)
    ]
    return prefix_layers, suffix_layers


def make_gradcam_heatmap(
    model: keras.Model,
    image_batch: tf.Tensor,
    class_index: int | None = None,
    target_layer_name: str | None = None,
    base_model_name_prefix: str = "resnet50",
) -> dict[str, Any]:
    """Create a Grad-CAM heatmap for a model with a nested application backbone."""
    if target_layer_name is None:
        base_model, target_layer_name = find_last_conv_layer(
            model,
            name_prefix=base_model_name_prefix,
        )
    else:
        base_model = find_nested_base_model(model, name_prefix=base_model_name_prefix)

    target_layer = base_model.get_layer(target_layer_name)
    activation_model = keras.Model(
        inputs=base_model.input,
        outputs=[target_layer.output, base_model.output],
    )
    prefix_layers, suffix_layers = _split_model_around_base(model, base_model)

    with tf.GradientTape() as tape:
        x = image_batch
        for layer in prefix_layers:
            x = _call_layer(layer, x, training=False)

        conv_outputs, base_outputs = activation_model(x, training=False)
        predictions = base_outputs
        for layer in suffix_layers:
            predictions = _call_layer(layer, predictions, training=False)

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]).numpy())

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients are None. Check target layer connectivity.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    if float(max_value.numpy()) > 0:
        heatmap = heatmap / max_value

    return {
        "heatmap": heatmap.numpy(),
        "predictions": predictions.numpy()[0],
        "class_index": class_index,
        "target_layer_name": target_layer_name,
        "base_model_name": base_model.name,
    }


def create_gradcam_overlay(
    image_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.35,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on a 0..1 RGB image."""
    image_array = np.asarray(image_array, dtype=np.float32)
    height, width = image_array.shape[:2]

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_image = Image.fromarray(heatmap_uint8).resize((width, height), Image.BILINEAR)
    heatmap_resized = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_resized)[..., :3].astype(np.float32)
    overlay = (1 - alpha) * image_array + alpha * colored_heatmap
    return np.clip(overlay, 0, 1)


def explain_image_with_gradcam(
    model: keras.Model,
    image_path: str | Path,
    class_names: list[str],
    class_index: int | None = None,
    target_layer_name: str | None = None,
    base_model_name_prefix: str = "resnet50",
    alpha: float = 0.35,
) -> dict[str, Any]:
    image_batch, image_array, runtime_path = load_image_for_model(image_path)
    gradcam = make_gradcam_heatmap(
        model=model,
        image_batch=image_batch,
        class_index=class_index,
        target_layer_name=target_layer_name,
        base_model_name_prefix=base_model_name_prefix,
    )

    predictions = gradcam["predictions"]
    predicted_index = int(np.argmax(predictions))
    explained_index = int(gradcam["class_index"])
    overlay = create_gradcam_overlay(image_array, gradcam["heatmap"], alpha=alpha)

    return {
        "image_path": runtime_path,
        "image_array": image_array,
        "heatmap": gradcam["heatmap"],
        "overlay": overlay,
        "predictions": predictions,
        "predicted_index": predicted_index,
        "predicted_class": class_names[predicted_index],
        "predicted_confidence": float(predictions[predicted_index]),
        "explained_index": explained_index,
        "explained_class": class_names[explained_index],
        "explained_confidence": float(predictions[explained_index]),
        "target_layer_name": gradcam["target_layer_name"],
        "base_model_name": gradcam["base_model_name"],
    }


def save_gradcam_figure(
    explanation: dict[str, Any],
    save_path: str | Path,
    true_class: str | None = None,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    title = (
        f"Predicted: {explanation['predicted_class']} "
        f"({explanation['predicted_confidence']:.3f})"
    )
    if true_class is not None:
        title = f"True: {true_class}\n{title}"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(explanation["image_array"])
    axes[0].set_title("Input")
    axes[1].imshow(explanation["heatmap"], cmap="jet")
    axes[1].set_title(f"Grad-CAM: {explanation['target_layer_name']}")
    axes[2].imshow(explanation["overlay"])
    axes[2].set_title(title)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_presentation_figure(
    explanation: dict[str, Any],
    save_path: str | Path,
    true_class: str | None = None,
    interpretation: str | None = None,
) -> Path:
    """Save a business-facing figure with input, overlay, and a plain-language caption."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    correct = true_class == explanation["predicted_class"] if true_class is not None else None
    caption = build_xai_caption(
        predicted_class=explanation["predicted_class"],
        confidence=explanation["predicted_confidence"],
        true_class=true_class,
        correct=correct,
    )
    if interpretation:
        caption = f"{caption}\n{interpretation}"

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4),
        gridspec_kw={"width_ratios": [1, 1, 1.15]},
    )
    axes[0].imshow(explanation["image_array"])
    axes[0].set_title("Original image")
    axes[1].imshow(explanation["overlay"])
    axes[1].set_title("Model attention")
    axes[2].text(0, 0.95, caption, va="top", ha="left", wrap=True, fontsize=11)
    axes[2].set_title("Plain-language note")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_failure_analysis_figure(
    predicted_explanation: dict[str, Any],
    true_class_explanation: dict[str, Any],
    save_path: str | Path,
    true_class: str,
) -> Path:
    """Save a figure comparing attention for predicted class vs true class."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    confidence = predicted_explanation["predicted_confidence"]
    risk_note = "High-confidence error: deployment risk." if confidence >= 0.95 else "Incorrect or uncertain case."
    caption = (
        f"Actual label: {format_class_name_for_business(true_class)}\n"
        f"Model prediction: {format_class_name_for_business(predicted_explanation['predicted_class'])}\n"
        f"Confidence: {confidence:.1%}\n"
        f"{risk_note}\n"
        "Compare the predicted-class attention with the true-class attention."
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(predicted_explanation["image_array"])
    axes[0].set_title("Original image")
    axes[1].imshow(predicted_explanation["overlay"])
    axes[1].set_title("Attention for prediction")
    axes[2].imshow(true_class_explanation["overlay"])
    axes[2].set_title("Attention for true class")
    axes[3].text(0, 0.95, caption, va="top", ha="left", wrap=True, fontsize=10)
    axes[3].set_title("Failure note")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_model_comparison_figure(
    image_array: np.ndarray,
    model_explanations: list[dict[str, Any]],
    save_path: str | Path,
    true_class: str,
) -> Path:
    """Save a side-by-side attention comparison for multiple models on one image."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(model_explanations)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))
    axes[0].imshow(image_array)
    axes[0].set_title(f"Actual: {format_class_name_for_business(true_class)}")
    axes[0].axis("off")

    for ax, explanation in zip(axes[1:], model_explanations):
        label = explanation.get("model_label", "Model")
        predicted = format_class_name_for_business(explanation["predicted_class"])
        confidence = explanation["predicted_confidence"]
        ax.imshow(explanation["overlay"])
        ax.set_title(f"{label}\n{predicted}\n{confidence:.1%}")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
