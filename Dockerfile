FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8001

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/task1 ./data/task1
COPY models/best_model.keras ./models/best_model.keras
COPY models/class_names.json ./models/class_names.json
COPY models/model_metadata.json ./models/model_metadata.json

RUN mkdir -p outputs/logs

EXPOSE 8001

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
