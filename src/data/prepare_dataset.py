from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    VALID_IMAGE_EXTENSIONS,
)


def count_direct_image_files(directory: Path) -> int:
    count = 0
    for ext in VALID_IMAGE_EXTENSIONS:
        count += len(list(directory.glob(f"*{ext}")))
        count += len(list(directory.glob(f"*{ext.upper()}")))
    return count


def find_dataset_root(start_path: Path, expected_classes: int | None = None) -> Path:
    """Find the directory whose direct children are class folders."""
    start_path = Path(start_path)
    candidates: list[tuple[Path, int]] = []

    directories = [start_path, *[p for p in start_path.rglob("*") if p.is_dir()]]
    for directory in directories:
        class_like_dirs = [
            p for p in directory.iterdir()
            if p.is_dir() and count_direct_image_files(p) > 0
        ]
        if class_like_dirs:
            candidates.append((directory, len(class_like_dirs)))

    if not candidates:
        raise ValueError(f"Could not find class folders under: {start_path}")

    if expected_classes is not None:
        exact_matches = [
            path for path, class_count in candidates
            if class_count == expected_classes
        ]
        if exact_matches:
            return exact_matches[0]

    return max(candidates, key=lambda item: item[1])[0]


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
    dataset_path = Path(dataset_path)

    for class_dir in list_class_directories(dataset_path):
        class_name = class_dir.name
        image_files = list_image_files(class_dir)

        for image_path in image_files:
            rows.append(
                {
                    "image_path": str(image_path),
                    "relative_path": str(image_path.relative_to(dataset_path)),
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


def targeted_oversample_dataframe(
    df: pd.DataFrame,
    target_min_count: int,
    class_column: str = "class_name",
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Oversample only classes below target_min_count by duplicating rows."""
    if target_min_count <= 0:
        raise ValueError("target_min_count must be positive")

    sampled_parts = []
    for class_name, class_df in df.groupby(class_column, sort=True):
        sampled_parts.append(class_df)

        needed = target_min_count - len(class_df)
        if needed > 0:
            sampled_parts.append(
                class_df.sample(
                    n=needed,
                    replace=True,
                    random_state=random_state,
                )
            )

    oversampled_df = pd.concat(sampled_parts, ignore_index=True)
    return oversampled_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


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


def save_class_names(class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
