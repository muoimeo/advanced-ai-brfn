from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import TASK1_DATA_DIR
from src.recommender.schemas import RecommenderDataset


REQUIRED_FILES = {
    "customers": "customers.csv",
    "producers": "producers.csv",
    "products": "products.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
}

REQUIRED_COLUMNS = {
    "customers": {"customer_id", "customer_type", "postcode_area"},
    "producers": {"producer_id", "producer_name", "postcode_area"},
    "products": {
        "product_id",
        "product_name",
        "category",
        "producer_id",
        "seasonal_start_month",
        "seasonal_end_month",
        "base_price",
    },
    "orders": {"order_id", "customer_id", "order_date", "total_amount"},
    "order_items": {"order_id", "product_id", "quantity", "unit_price"},
}

FORBIDDEN_CUSTOMER_COLUMNS = {
    "name",
    "first_name",
    "last_name",
    "email",
    "phone",
    "telephone",
    "address",
    "full_address",
    "delivery_address",
}


class Task1DataError(ValueError):
    """Raised when the Task 1 DESD export is missing or invalid."""


def validate_required_files(data_dir: Path = TASK1_DATA_DIR) -> None:
    missing = [
        filename
        for filename in REQUIRED_FILES.values()
        if not (data_dir / filename).exists()
    ]
    if missing:
        raise Task1DataError(
            "Task 1 DESD seed export is missing required files: "
            + ", ".join(missing)
            + f". Expected folder: {data_dir}"
        )


def validate_columns(name: str, dataframe: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS[name] - set(dataframe.columns)
    if missing:
        raise Task1DataError(
            f"{name}.csv is missing required columns: {sorted(missing)}"
        )


def validate_privacy_safe_columns(customers: pd.DataFrame) -> None:
    present = {column.lower() for column in customers.columns}
    forbidden = sorted(FORBIDDEN_CUSTOMER_COLUMNS & present)
    if forbidden:
        raise Task1DataError(
            "customers.csv contains non-anonymised or unnecessary personal columns: "
            + ", ".join(forbidden)
        )


def load_task1_dataset(data_dir: Path = TASK1_DATA_DIR) -> RecommenderDataset:
    data_dir = Path(data_dir)
    validate_required_files(data_dir)
    loaded = {
        name: pd.read_csv(data_dir / filename)
        for name, filename in REQUIRED_FILES.items()
    }

    for name, dataframe in loaded.items():
        validate_columns(name, dataframe)
    validate_privacy_safe_columns(loaded["customers"])

    loaded["orders"]["order_date"] = pd.to_datetime(
        loaded["orders"]["order_date"],
        errors="coerce",
    )
    if loaded["orders"]["order_date"].isna().any():
        raise Task1DataError("orders.csv contains invalid order_date values.")

    for column in ["seasonal_start_month", "seasonal_end_month"]:
        loaded["products"][column] = pd.to_numeric(
            loaded["products"][column],
            errors="coerce",
        ).astype("Int64")
    for column in ["base_price"]:
        loaded["products"][column] = pd.to_numeric(
            loaded["products"][column],
            errors="coerce",
        )
    for column in ["quantity", "unit_price"]:
        loaded["order_items"][column] = pd.to_numeric(
            loaded["order_items"][column],
            errors="coerce",
        )

    return RecommenderDataset(
        customers=loaded["customers"],
        producers=loaded["producers"],
        products=loaded["products"],
        orders=loaded["orders"],
        order_items=loaded["order_items"],
        data_dir=data_dir,
    )


def build_order_lines(dataset: RecommenderDataset) -> pd.DataFrame:
    lines = dataset.order_items.merge(
        dataset.orders[["order_id", "customer_id", "order_date", "total_amount"]],
        on="order_id",
        how="inner",
    )
    lines = lines.merge(dataset.products, on="product_id", how="left")
    lines = lines.merge(
        dataset.customers[["customer_id", "customer_type"]],
        on="customer_id",
        how="left",
    )
    lines = lines.merge(
        dataset.producers[["producer_id", "producer_name"]],
        on="producer_id",
        how="left",
    )
    return lines


def dataset_summary(dataset: RecommenderDataset, order_lines: pd.DataFrame) -> dict:
    if dataset.orders.empty:
        date_range = {"start": None, "end": None}
    else:
        date_range = {
            "start": dataset.orders["order_date"].min().date().isoformat(),
            "end": dataset.orders["order_date"].max().date().isoformat(),
        }

    return {
        "dataset_type": "synthetic_desd_seed_export",
        "customers": int(dataset.customers["customer_id"].nunique()),
        "producers": int(dataset.producers["producer_id"].nunique()),
        "products": int(dataset.products["product_id"].nunique()),
        "orders": int(dataset.orders["order_id"].nunique()),
        "order_lines": int(len(order_lines)),
        "date_range": date_range,
        "source_folder": str(dataset.data_dir),
        "limitations": [
            "synthetic_seed_data_not_real_customer_behaviour",
            "metrics_are_proof_of_concept_not_production_accuracy",
        ],
    }
