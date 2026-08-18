import os
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://fraud_user:fraud_password@postgres:5432/fraud_db"
)

def load_data_from_db(table_name: str = "raw_transactions") -> pd.DataFrame:
    """Fetch raw transactions dataset from PostgreSQL."""
    logging.info(f"Fetching raw data from PostgreSQL table: '{table_name}'...")
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
    logging.info(f"Successfully loaded {len(df)} rows from database.")
    return df

def validate_data(df: pd.DataFrame) -> None:
    """Perform basic data checks (missing values, schema, target distribution)."""
    logging.info("Starting data validation checks...")
    
    # Check missing values
    missing_counts = df.isnull().sum().sum()
    if missing_counts > 0:
        logging.warning(f"Found {missing_counts} missing values in the dataset.")
    else:
        logging.info("Zero missing values detected.")

    # Check class distribution
    if "Class" in df.columns:
        fraud_count = df["Class"].sum()
        total_count = len(df)
        fraud_ratio = (fraud_count / total_count) * 100
        logging.info(f"Target distribution -> Normal: {total_count - fraud_count}, Fraud: {fraud_count} ({fraud_ratio:.3f}%)")

def process_and_split_data(df: pd.DataFrame, output_dir: str = "data/processed") -> None:
    """
    Apply RobustScaler on Time/Amount features and split data into train, val, and test sets.
    Prevents data leakage by fitting scalers strictly on the training set.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # First split: Train (70%) vs Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Second split: Val (15%) vs Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logging.info(f"Dataset split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Fit RobustScaler ONLY on training data to prevent data leakage
    scaler_amount = RobustScaler()
    scaler_time = RobustScaler()

    X_train["scaled_amount"] = scaler_amount.fit_transform(X_train[["Amount"]])
    X_train["scaled_time"] = scaler_time.fit_transform(X_train[["Time"]])

    X_val["scaled_amount"] = scaler_amount.transform(X_val[["Amount"]])
    X_val["scaled_time"] = scaler_time.transform(X_val[["Time"]])

    X_test["scaled_amount"] = scaler_amount.transform(X_test[["Amount"]])
    X_test["scaled_time"] = scaler_time.transform(X_test[["Time"]])

    # Drop original Time and Amount columns
    columns_to_drop = ["Time", "Amount"]
    X_train = X_train.drop(columns=columns_to_drop)
    X_val = X_val.drop(columns=columns_to_drop)
    X_test = X_test.drop(columns=columns_to_drop)

    # Recombine features and target for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Export to processed directory
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    logging.info(f"Processed datasets successfully saved to '{output_dir}/'.")

if __name__ == "__main__":
    df_raw = load_data_from_db()
    validate_data(df_raw)
    process_and_split_data(df_raw)