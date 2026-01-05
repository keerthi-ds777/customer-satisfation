import mlflow
from zenml import step
import pandas as pd
import logging
from pydantic import BaseModel
from src.data_cleaning import IngestData
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=True, experiment_tracker=experiment_tracker.name)
def ingesting(data_path:str) -> pd.DataFrame:
    """
    Ingest data from a CSV file.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the ingested data.
    """
    # Initialize the IngestData class with the provided data path
    try:
        ingest_data = IngestData(data_path=data_path)
        df = ingest_data.get_data()
        if df is None:
            logging.error(f"IngestData.get_data() returned None for {data_path}. This indicates a problem within the IngestData class or data accessibility.")
            raise ValueError("Ingesting data failed: get_data() returned None.")
        logging.info(f"Successfully ingested data from {data_path}")
        return df
    except Exception as e:
        logging.error(f"Error ingesting data from {data_path}: {e}")
        raise ValueError(f"Failed to ingest data from {data_path}: {e}")
if __name__ == "__main__":
    # Example usage
    data_path = "data/olist_customers_dataset.csv"
    df = ingesting(data_path=data_path)
    print(df.head())