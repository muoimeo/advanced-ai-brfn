from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMAGE_SIZE, NUM_CLASSES


def build_mobilenetv2_model(
    num_classes: int = NUM_CLASSES,
    augmentation: keras.Model | None = None,
    dropout_rate: float = 0.3,
    train_base: bool = False,
    weights: str | None = "imagenet",
) -> keras.Model:
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    x = augmentation(inputs) if augmentation is not None else inputs
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenetv2_preprocess")(x)

    base_model = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = train_base

    # Keep BatchNorm statistics stable for both frozen training and light fine-tuning.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name="mobilenetv2_transfer")


def unfreeze_mobilenetv2_top(
    model: keras.Model,
    trainable_layers: int = 30,
) -> keras.Model:
    base_model_candidates = [
        layer for layer in model.layers
        if isinstance(layer, keras.Model) and layer.name.startswith("mobilenetv2")
    ]

    if not base_model_candidates:
        raise ValueError("Could not find a nested MobileNetV2 base model layer.")

    base_model = base_model_candidates[0]
    base_model.trainable = True

    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    return model
