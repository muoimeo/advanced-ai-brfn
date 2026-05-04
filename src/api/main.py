from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status

from src.config import LOGS_DIR
from src.inference.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    MonitoringSummaryResponse,
    PredictionResponse,
    ReorderResponse,
)
from src.monitoring.feedback_monitoring import build_feedback_monitoring_summary
from src.recommender.data_loader import Task1DataError
from src.recommender.pipeline import get_reorder_recommendations
from src.recommender.quick_reorder import METHOD_FREQUENCY_RECENCY, SUPPORTED_METHODS

try:
    from src.inference.predictor import get_metadata, get_model, predict_image_bytes
except Exception as predictor_import_error:
    _PREDICTOR_IMPORT_ERROR = predictor_import_error

    def get_model():
        raise RuntimeError("Computer-vision predictor is unavailable.") from _PREDICTOR_IMPORT_ERROR

    def predict_image_bytes(image_bytes: bytes) -> dict:
        raise RuntimeError("Computer-vision predictor is unavailable.") from _PREDICTOR_IMPORT_ERROR

    def get_metadata() -> dict:
        return {
            "model_name": None,
            "predictor_available": False,
            "error": str(_PREDICTOR_IMPORT_ERROR),
        }


ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png"}


app = FastAPI(
    title="BRFN Freshness Classification API",
    version="0.1.0",
)


def _json_safe_prediction_record(prediction: dict, content_type: str | None) -> dict:
    return {
        "prediction_id": prediction.get("prediction_id"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": prediction.get("model_name"),
        "model_version": prediction.get("model_version"),
        "predicted_class": prediction.get("predicted_class"),
        "freshness_status": prediction.get("freshness_status"),
        "confidence": prediction.get("confidence"),
        "top1_top2_margin": prediction.get("top1_top2_margin"),
        "quality_grade": prediction.get("quality_grade"),
        "recommended_action": prediction.get("recommended_action"),
        "quality": prediction.get("quality", {}),
        "reason_codes": prediction.get("reason_codes", []),
        "manual_review_required": prediction.get("manual_review_required"),
        "content_type": content_type,
    }


def _append_jsonl(filename: str, record: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / filename
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


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
    if file.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Upload a JPEG or PNG image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty.",
        )

    try:
        prediction = predict_image_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode or classify the uploaded image.",
        ) from exc

    prediction["prediction_id"] = str(uuid4())
    _append_jsonl(
        "predictions.jsonl",
        _json_safe_prediction_record(prediction, file.content_type),
    )
    return PredictionResponse(**prediction)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    record = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    record["created_at"] = datetime.now().isoformat(timespec="seconds")
    _append_jsonl("feedback.jsonl", record)

    return FeedbackResponse(status="logged")


@app.get("/monitoring/feedback-summary", response_model=MonitoringSummaryResponse)
def feedback_monitoring_summary() -> MonitoringSummaryResponse:
    return MonitoringSummaryResponse(**build_feedback_monitoring_summary(LOGS_DIR))


@app.get(
    "/recommend/reorder",
    response_model=ReorderResponse,
    response_model_exclude_none=True,
)
def recommend_reorder(
    customer_id: str,
    top_k: int = Query(3, ge=1, le=20),
    method: str = METHOD_FREQUENCY_RECENCY,
    include_discovery: bool = False,
) -> ReorderResponse:
    if method not in SUPPORTED_METHODS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported recommender method: {method}",
        )

    try:
        response = get_reorder_recommendations(
            customer_id=customer_id,
            top_k=top_k,
            method=method,
            include_discovery=include_discovery,
        )
    except Task1DataError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return ReorderResponse(**response)
