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
