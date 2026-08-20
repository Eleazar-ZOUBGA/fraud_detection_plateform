import os
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set MLflow tracking URI from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("fraud_detection_baseline")


def load_processed_data(data_dir: str = "data/processed"):
    """Load train, val, and test datasets from CSV files."""
    logging.info(f"Loading datasets from {data_dir}...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    
    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]
    
    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"]
    
    return X_train, y_train, X_val, y_val


def evaluate_model(model, X_val, y_val):
    """Compute key evaluation metrics suited for imbalanced classification."""
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    # Compute PR-AUC (Precision-Recall Area Under Curve)
    precision_series, recall_series, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall_series, precision_series)

    metrics = {
        "pr_auc": float(pr_auc),
        "recall": float(recall_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
    }
    
    cm = confusion_matrix(y_val, y_pred)
    return metrics, cm


def train_baseline():
    """Train baseline Logistic Regression model and log metrics to MLflow."""
    X_train, y_train, X_val, y_val = load_processed_data()

    params = {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    }

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        logging.info("Training Logistic Regression baseline...")
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(params)

        # Evaluate model
        metrics, cm = evaluate_model(model, X_val, y_val)
        logging.info(f"Validation Metrics -> PR-AUC: {metrics['pr_auc']:.4f}, Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log confusion matrix values
        mlflow.log_metric("tn", float(cm[0, 0]))
        mlflow.log_metric("fp", float(cm[0, 1]))
        mlflow.log_metric("fn", float(cm[1, 0]))
        mlflow.log_metric("tp", float(cm[1, 1]))

        # Log model artifact with name parameter
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        logging.info("Baseline model training completed and logged to MLflow successfully.")

def train_random_forest():
    """Train Random Forest model and log metrics to MLflow."""
    X_train, y_train, X_val, y_val = load_processed_data()

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="random_forest_balanced"):
        logging.info("Training Random Forest model...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(params)

        # Evaluate model
        metrics, cm = evaluate_model(model, X_val, y_val)
        logging.info(f"Validation Metrics -> PR-AUC: {metrics['pr_auc']:.4f}, Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log confusion matrix values
        mlflow.log_metric("tn", float(cm[0, 0]))
        mlflow.log_metric("fp", float(cm[0, 1]))
        mlflow.log_metric("fn", float(cm[1, 0]))
        mlflow.log_metric("tp", float(cm[1, 1]))

        # Log model artifact
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        logging.info("Random Forest training completed and logged to MLflow successfully.")

def train_xgboost():
    """Train XGBoost model with scale_pos_weight for class imbalance and log to MLflow."""
    X_train, y_train, X_val, y_val = load_processed_data()

    # Calculate scale_pos_weight (ratio of negative to positive samples)
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)

    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": ratio,
        "random_state": 42,
        "eval_metric": "logloss",
    }

    with mlflow.start_run(run_name="xgboost_scaled"):
        logging.info("Training XGBoost model...")
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(params)

        # Evaluate model
        metrics, cm = evaluate_model(model, X_val, y_val)
        logging.info(f"Validation Metrics -> PR-AUC: {metrics['pr_auc']:.4f}, Recall: {metrics['recall']:.4f}, Precision: {metrics['precision']:.4f}")

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log confusion matrix values
        mlflow.log_metric("tn", float(cm[0, 0]))
        mlflow.log_metric("fp", float(cm[0, 1]))
        mlflow.log_metric("fn", float(cm[1, 0]))
        mlflow.log_metric("tp", float(cm[1, 1]))

        # Log model artifact with registration inside the run
        result = mlflow.xgboost.log_model(
            xgb_model=model, 
            artifact_path="model",
            registered_model_name="FraudDetectionModel"
        )
        logging.info("XGBoost training completed and registered in MLflow Model Registry.")

if __name__ == "__main__":
    train_baseline()
    train_random_forest()
    train_xgboost()