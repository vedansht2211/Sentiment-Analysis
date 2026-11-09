import os
import re
import html
from contextlib import asynccontextmanager
from typing import Callable

import mlflow
import mlflow.transformers
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import pipeline

# MLFlow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = "sentiment-classifier"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@champion"

# "registry" = load from MLflow, "local" = load from the models folder
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "registry")
LOCAL_MODEL_PATH = "models/sentiment_bert"
MAX_LEN = 64


class SentimentFeatures(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


# same cleaning as training
def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_mlflow_model():
    if MODEL_SOURCE == "local":
        return pipeline("text-classification",
                        model=LOCAL_MODEL_PATH, tokenizer=LOCAL_MODEL_PATH)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.transformers.load_model(MODEL_URI)   # returns a pipeline


def load_model_version():
    if MODEL_SOURCE == "local":
        return "local"
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME, alias="champion"
    ).version


def create_app(model_loader: Callable = load_mlflow_model,
               version_loader: Callable = load_model_version):

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
        except Exception as ex:
            raise RuntimeError(f"Failed to load model: {ex}")
        app.state.model_version = version_loader()
        yield

    app = FastAPI(
        title="Sentiment Classification API",
        description="Sentiment Analysis API powered by MLFlow",
        version="3.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    def ready_to_serve():
        return {"status": "ready to serve..."}

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "model_name": REGISTERED_MODEL_NAME,
            "model_version": app.state.model_version
        }

    @app.post("/predict")
    def predict(data: SentimentFeatures, request: Request):
        model = request.app.state.model
        text = clean_text(data.text)
        if not text:
            raise HTTPException(status_code=422, detail="Text is empty after cleaning")
        out = model(text, truncation=True, max_length=MAX_LEN)[0]
        return {
            "class_name": out["label"],
            "confidence": round(out["score"], 4),
        }

    return app


# Production application instance
app = create_app()