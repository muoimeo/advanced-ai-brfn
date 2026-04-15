from __future__ import annotations

import pandas as pd
import tensorflow as tf

from src.config import IMAGE_SIZE, BATCH_SIZE


def build_class_name_mapping(df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    class_names = sorted(df["class_name"].unique().tolist())
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    return class_names, class_to_index


def decode_and_resize_image(image_path: tf.Tensor) -> tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def make_dataset_from_dataframe(
    df: pd.DataFrame,
    class_to_index: dict[str, int],
    training: bool = False,
) -> tf.data.Dataset:
    image_paths = df["image_path"].tolist()
    labels = [class_to_index[name] for name in df["class_name"].tolist()]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def _load_example(path, label):
        image = decode_and_resize_image(path)
        return image, label

    ds = ds.map(_load_example, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds