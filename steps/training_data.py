from zenml import step
import pandas as pd
import logging
from src.data_cleaning import DataCleaning, ModelTrainig
from zenml.client import Client
from typing import NamedTuple
import numpy as np
import mlflow
from sklearn.base import ClassifierMixin
experiment_tracker = Client().active_stack.experiment_tracker
from typing import Tuple
import numpy as np


class ModelOutput(NamedTuple):
    model: object  # e.g., sklearn.base.BaseEstimator
    y_pred: np.ndarray


#@step(experiment_tracker=experiment_tracker.name)
@step(enable_cache=True, experiment_tracker=experiment_tracker.name)
def train_model(x_train, x_test, y_train, y_test) -> Tuple[ClassifierMixin, np.ndarray]:
    model_trainer = ModelTrainig(x_train, x_test, y_train, y_test)
    trained_model = model_trainer.fit_model()
    
    y_pred = trained_model.predict(x_test)

    # Log model parameters
    try:
        params = trained_model.get_params()
        mlflow.log_params(params)
    except Exception as e:
        logging.warning(f"Could not log parameters: {e}")

        # ✅ Automatically logs the model AND registers it
    mlflow.sklearn.log_model(trained_model, name="model", registered_model_name="rf_model")

    return trained_model, y_pred
    
if __name__ == "__main__":
    data = pd.read_csv("data/olist_customers_dataset.csv")
    data_cleaning = DataCleaning(data)
    x, y = data_cleaning.preprocess()
    model = train_model(x, x, y, y)
    print(model)

    print(model.y_pred)