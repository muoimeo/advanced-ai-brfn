from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RecommenderDataset:
    customers: pd.DataFrame
    producers: pd.DataFrame
    products: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    data_dir: Path


@dataclass
class Recommendation:
    customer_id: str
    customer_type: str | None
    method: str
    rank: int
    product_id: str
    product_name: str
    producer_id: str
    score: float
    reason_codes: list[str] = field(default_factory=list)
    reason_text: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["score"] = round(float(data["score"]), 6)
        data["reason_codes"] = "|".join(self.reason_codes)
        return data

    def to_api_dict(self) -> dict:
        data = asdict(self)
        data["score"] = round(float(data["score"]), 6)
        return data
