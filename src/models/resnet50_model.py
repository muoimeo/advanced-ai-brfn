from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMAGE_SIZE, NUM_CLASSES


@keras.utils.register_keras_serializable(package="ai_system")
class ResNet50Preprocess(layers.Layer):
    """Preprocess 0..1 RGB images for Keras ResNet50 pretrained weights."""

    def call(self, inputs):
        x = inputs * 255.0
        x = x[..., ::-1]  # RGB to BGR
        mean = tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)
        return x - mean

    def get_config(self):
        return super().get_config()


def build_resnet50_model(
    num_classes: int = NUM_CLASSES,
    augmentation: keras.Model | None = None,
    dropout_rate: float = 0.4,
    train_base: bool = False,
    weights: str | None = "imagenet",
) -> keras.Model:
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    x = augmentation(inputs) if augmentation is not None else inputs
    x = ResNet50Preprocess(name="resnet50_preprocess")(x)

    base_model = keras.applications.ResNet50(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = train_base

    # Keep BatchNorm statistics stable for frozen training and light fine-tuning.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name="resnet50_transfer")


def unfreeze_resnet50_top(
    model: keras.Model,
    trainable_layers: int = 30,
) -> keras.Model:
    base_model_candidates = [
        layer for layer in model.layers
        if isinstance(layer, keras.Model) and layer.name.startswith("resnet50")
    ]

    if not base_model_candidates:
        raise ValueError("Could not find a nested ResNet50 base model layer.")

    base_model = base_model_candidates[0]
    base_model.trainable = True

    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    return model
