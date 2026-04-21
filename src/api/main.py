from __future__ import annotations

import json
from datetime import datetime

from fastapi import FastAPI, File, UploadFile

from src.config import LOGS_DIR
from src.inference.predictor import get_metadata, get_model, predict_image_bytes
from src.inference.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)


app = FastAPI(
    title="BRFN Freshness Classification API",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    metadata = get_metadata()
    return ModelInfoResponse(
        model_name=metadata.get("model_name"),
        base_model_family=metadata.get("base_model_family"),
        num_classes=metadata.get("num_classes"),
        image_size=metadata.get("image_size"),
        metrics=metadata.get("metrics"),
        metadata=metadata,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    image_bytes = await file.read()
    prediction = predict_image_bytes(image_bytes)
    return PredictionResponse(**prediction)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "feedback.jsonl"
    record = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    record["created_at"] = datetime.now().isoformat(timespec="seconds")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return FeedbackResponse(status="logged")
