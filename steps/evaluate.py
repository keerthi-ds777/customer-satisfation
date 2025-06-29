from zenml import step
from zenml.client import Client
from typing import Tuple
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import mlflow
experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def evaluate_model(y_test, y_pred) -> Tuple[float, float, float, float]:
 

    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    return accuracy, mse, r2, mae