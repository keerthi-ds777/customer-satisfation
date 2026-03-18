from zenml import step
import pandas as pd
import logging
from src.data_cleaning import DataCleaning
from src.model_training import ModelTrainig
from zenml.client import Client
from typing import NamedTuple
import numpy as np
import mlflow
from sklearn.base import ClassifierMixin
experiment_tracker = Client().active_stack.experiment_tracker
from typing import Tuple
import numpy as np


 


#@step(experiment_tracker=experiment_tracker.name)
@step(enable_cache=True, experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    registerd_model: bool = False,
    model_name: str = "rf_model",
    model_version: str = "1",
    log_model: bool = False,
) -> Tuple[ClassifierMixin, np.ndarray]:
    model_trainer = ModelTrainig(x_train, x_test, y_train, y_test)
    trained_model, y_pred = model_trainer.fit_model(
        registerd_model=registerd_model,
        model_name=model_name,
        model_version=model_version,
        log_model=log_model
    )

    # Log model parameters
    try:
        params = trained_model.get_params()
        mlflow.log_params(params)
    except Exception as e:
        logging.warning(f"Could not log parameters: {e}")

    return trained_model, y_pred
    
if __name__ == "__main__":
    data = pd.read_csv("data/olist_customers_dataset.csv")
    data_cleaning = DataCleaning(data)
    x, y = data_cleaning.preprocess()
    model = train_model(x,y)
    print(model)

    print(model.y_pred)