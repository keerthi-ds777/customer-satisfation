import logging
from zenml import pipeline
from steps.ingest_data import ingesting
from src.data_cleaning import DataCleaning, ModelTrainig
from steps.preprocessing import cleaning
from steps.training_data import train_model
from steps.evaluate import evaluate_model

@pipeline(enable_cache=False)
def run_pipeline(data_path: str) -> None:
    """
    Run the entire pipeline.

    Args:
        data_path (str): Path to the CSV file.
    """
    # Ingest data
    data = ingesting(data_path=data_path)
    
    # Clean data
    #preprocessing the data
    x_train, x_test, y_train, y_test = cleaning(data)  #
    
    # Train model
    model,y_pred=train_model(x_train, x_test, y_train, y_test) #train the model
    
    # Evaluate model
    accuracy, mse, r2, mae=evaluate_model(y_test,y_pred)
    logging.info(f"Model Accuracy: {accuracy}")
    logging.info(f"Model MSE: {mse}")
    logging.info(f"Model R2: {r2}")
    logging.info(f"Model MAE: {mae}")
