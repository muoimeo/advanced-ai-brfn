from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import TASK1_DATA_DIR, TASK1_OUTPUT_DIR
from src.recommender.data_loader import build_order_lines, load_task1_dataset
from src.recommender.evaluation import export_task1_outputs
from src.recommender.quick_reorder import METHOD_FREQUENCY_RECENCY, recommend


TASK1_LIMITATIONS = [
    "recommendations_based_on_synthetic_seed_data",
    "not_production_customer_behaviour",
]


def get_reorder_recommendations(
    customer_id: str,
    top_k: int = 3,
    method: str = METHOD_FREQUENCY_RECENCY,
    data_dir: Path = TASK1_DATA_DIR,
) -> dict:
    dataset = load_task1_dataset(data_dir)
    order_lines = build_order_lines(dataset)
    recommendation_date = (
        order_lines["order_date"].max() + pd.Timedelta(days=1)
        if not order_lines.empty
        else pd.Timestamp.today().normalize()
    )
    recommendations = recommend(
        customer_id=customer_id,
        order_lines=order_lines,
        products=dataset.products,
        method=method,
        top_k=top_k,
        recommendation_date=recommendation_date,
    )
    customer_rows = dataset.customers[dataset.customers["customer_id"] == customer_id]
    if not customer_rows.empty:
        customer_type = str(customer_rows.iloc[0]["customer_type"])
        for item in recommendations:
            if item.customer_type is None:
                item.customer_type = customer_type
    return {
        "customer_id": customer_id,
        "recommendation_date": recommendation_date.date().isoformat(),
        "method": method,
        "top_k": top_k,
        "recommendations": [item.to_api_dict() for item in recommendations],
        "limitations": TASK1_LIMITATIONS,
    }


def run_task1_evaluation(
    data_dir: Path = TASK1_DATA_DIR,
    output_dir: Path = TASK1_OUTPUT_DIR,
    top_k: int = 3,
) -> dict[str, str]:
    dataset = load_task1_dataset(data_dir)
    return export_task1_outputs(dataset, output_dir, top_k=top_k)


if __name__ == "__main__":
    written_files = run_task1_evaluation()
    for filename, path in written_files.items():
        print(f"{filename}: {path}")
