import os
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud detection service using XGBoost and MLflow.",
    version="1.0.0"
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None


class TransactionData(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float


def fetch_latest_model():
    """Fetch the latest registered XGBoost model from MLflow."""
    global model
    try:
        client = MlflowClient()
        latest_versions = client.get_latest_versions("FraudDetectionModel")
        if not latest_versions:
            logging.warning("No registered model version found for 'FraudDetectionModel'.")
            return
        
        latest_version = latest_versions[0].version
        model_uri = f"models:/FraudDetectionModel/{latest_version}"
        logging.info(f"Loading registered model version {latest_version} from {model_uri}...")
        model = mlflow.xgboost.load_model(model_uri)
        logging.info("Model loaded successfully into FastAPI application.")
    except Exception as e:
        logging.warning(f"Failed to load model from MLflow registry: {str(e)}")


@app.on_event("startup")
def startup_event():
    fetch_latest_model()


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(data: TransactionData):
    """Predict whether a transaction is fraudulent or legitimate."""
    global model
    if model is None:
        fetch_latest_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not available or registered yet.")

    try:
        df_input = pd.DataFrame([data.features])
        proba = float(model.predict_proba(df_input)[0, 1])
        prediction = bool(proba >= 0.5)

        return PredictionResponse(
            is_fraud=prediction,
            fraud_probability=round(proba, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")