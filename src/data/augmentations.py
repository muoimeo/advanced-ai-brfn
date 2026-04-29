from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_training_augmentation() -> keras.Sequential:
    """Image augmentation used only for training data."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            layers.RandomContrast(0.10),
        ],
        name="training_augmentation",
    )


def build_xai_corrected_augmentation() -> keras.Sequential:
    """Stronger augmentation for XAI-guided shortcut reduction experiments.

    The goal is to make the model less dependent on fixed backgrounds, lighting,
    and simple color/texture shortcuts while keeping produce identity visible.
    """
    augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(height_factor=(-0.08, 0.12), width_factor=(-0.08, 0.12)),
        layers.RandomTranslation(height_factor=0.06, width_factor=0.06),
        layers.RandomContrast(0.12),
    ]

    augmentation_layers.append(layers.GaussianNoise(0.015))

    return keras.Sequential(
        augmentation_layers,
        name="xai_corrected_augmentation",
    )
