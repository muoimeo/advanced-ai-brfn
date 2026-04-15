from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    VALID_IMAGE_EXTENSIONS,
)


def list_class_directories(dataset_path: Path) -> list[Path]:
    return sorted([p for p in dataset_path.iterdir() if p.is_dir()])


def list_image_files(class_dir: Path) -> list[Path]:
    image_files = []
    for ext in VALID_IMAGE_EXTENSIONS:
        image_files.extend(class_dir.glob(f"*{ext}"))
        image_files.extend(class_dir.glob(f"*{ext.upper()}"))
    return sorted(set(image_files))


def build_image_dataframe(dataset_path: Path) -> pd.DataFrame:
    rows = []

    for class_dir in list_class_directories(dataset_path):
        class_name = class_dir.name
        image_files = list_image_files(class_dir)

        for image_path in image_files:
            rows.append(
                {
                    "image_path": str(image_path),
                    "class_name": class_name,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No images found under dataset path: {dataset_path}")

    return df


def stratified_split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_RATIO),
        stratify=df["class_name"],
        random_state=RANDOM_SEED,
    )

    val_relative_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative_ratio),
        stratify=temp_df["class_name"],
        random_state=RANDOM_SEED,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split_csvs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)