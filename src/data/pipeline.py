import os
import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://fraud_user:fraud_password@postgres:5432/fraud_db"
)

def load_raw_data(csv_path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load the raw dataset from the CSV file.."""
    logging.info(f"Loading the CSV file from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file {csv_path} is not found. Make sure to place it in data/")
    
    df = pd.read_csv(csv_path)
    logging.info(f"Dataset charged successfully. Dimensions : {df.shape}")
    return df

def ingest_to_postgres(df: pd.DataFrame, table_name: str = "raw_transactions") -> None:
    """Write the raw DataFrame to the PostgreSQL database."""
    logging.info(f"Connecting to PostgreSQL and writing to the table '{table_name}'...")
    engine = create_engine(DATABASE_URL)
    
    # Write by chunks to optimize memory if the dataset is large
    df.to_sql(table_name, con=engine, if_exists="replace", index=False, chunksize=10000)
    logging.info("Ingestion of data into PostgreSQL completed successfully.")

if __name__ == "__main__":
    df = load_raw_data()
    ingest_to_postgres(df)