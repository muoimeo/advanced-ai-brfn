from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

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


AUGMENTATION_PREFIX_PATTERNS = (
    re.compile(r"^rotated_by_\d+_", flags=re.IGNORECASE),
    re.compile(r"^vertical_flip_", flags=re.IGNORECASE),
    re.compile(r"^horizontal_flip_", flags=re.IGNORECASE),
    re.compile(r"^translation_", flags=re.IGNORECASE),
    re.compile(r"^saltandpepper_", flags=re.IGNORECASE),
    re.compile(r"^noise_", flags=re.IGNORECASE),
    re.compile(r"^brightness_", flags=re.IGNORECASE),
    re.compile(r"^contrast_", flags=re.IGNORECASE),
)
AUGMENTATION_SUFFIX_PATTERNS = (
    re.compile(r"_augmented_\d+(?=\.)", flags=re.IGNORECASE),
)


def source_image_id_from_relative_path(relative_path: str) -> str:
    """Normalize known offline-augmentation filenames to a source-image id."""
    path_text = str(relative_path).replace("\\", "/")
    if "/" not in path_text:
        class_name = ""
        filename = path_text
    else:
        class_name, filename = path_text.split("/", maxsplit=1)

    normalized_filename = filename
    for pattern in AUGMENTATION_PREFIX_PATTERNS:
        normalized_filename = pattern.sub("", normalized_filename)
    for pattern in AUGMENTATION_SUFFIX_PATTERNS:
        normalized_filename = pattern.sub("", normalized_filename)

    return f"{class_name}/{normalized_filename}" if class_name else normalized_filename


def add_source_image_id(
    df: pd.DataFrame,
    relative_path_column: str = "relative_path",
) -> pd.DataFrame:
    """Return a copy with source_image_id for leakage auditing/grouped splitting."""
    if relative_path_column not in df.columns:
        raise KeyError(f"Missing required column: {relative_path_column}")

    output_df = df.copy()
    output_df["source_image_id"] = output_df[relative_path_column].map(
        source_image_id_from_relative_path
    )
    return output_df


def audit_source_image_overlap(
    split_frames: dict[str, pd.DataFrame],
    relative_path_column: str = "relative_path",
) -> pd.DataFrame:
    """Find normalized source-image identities that appear in more than one split."""
    rows = []
    for split_name, split_df in split_frames.items():
        audited_df = add_source_image_id(split_df, relative_path_column=relative_path_column)
        for _, row in audited_df.iterrows():
            rows.append(
                {
                    "split": split_name,
                    "class_name": row.get("class_name"),
                    "relative_path": row.get(relative_path_column),
                    "source_image_id": row["source_image_id"],
                }
            )

    combined = pd.DataFrame(rows)
    if combined.empty:
        return combined

    split_counts = (
        combined.groupby("source_image_id")["split"]
        .nunique()
        .rename("split_count")
        .reset_index()
    )
    overlapping_ids = split_counts.loc[split_counts["split_count"] > 1, "source_image_id"]
    return (
        combined[combined["source_image_id"].isin(overlapping_ids)]
        .sort_values(["source_image_id", "split", "relative_path"])
        .reset_index(drop=True)
    )


def grouped_stratified_split_dataframe(
    df: pd.DataFrame,
    group_column: str = "source_image_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits while keeping source-image groups together.

    This is intended for datasets that already include offline augmented variants.
    """
    if group_column not in df.columns:
        df = add_source_image_id(df)

    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    df = df.reset_index(drop=True)
    n_splits = max(2, round(1.0 / (VAL_RATIO + TEST_RATIO)))
    outer_splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    train_index, temp_index = next(
        outer_splitter.split(df, y=df["class_name"], groups=df[group_column])
    )

    train_df = df.iloc[train_index].reset_index(drop=True)
    temp_df = df.iloc[temp_index].reset_index(drop=True)

    inner_splitter = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    val_index, test_index = next(
        inner_splitter.split(
            temp_df,
            y=temp_df["class_name"],
            groups=temp_df[group_column],
        )
    )

    val_df = temp_df.iloc[val_index].reset_index(drop=True)
    test_df = temp_df.iloc[test_index].reset_index(drop=True)
    return train_df, val_df, test_df


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
