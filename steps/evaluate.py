from zenml import step
from zenml.client import Client
from typing import Tuple
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import mlflow
from typing import Annotated
experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=True, experiment_tracker=experiment_tracker.name)
def evaluate_model(y_test, y_pred) -> Tuple[Annotated[float,'Accuracy'],
                                            Annotated[float,'MSE'],
                                            Annotated[float,'R2'],
                                            Annotated[float,'MAE']]:
 

            accuracy = accuracy_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            mlflow.log_metric("accuracy", float(accuracy))
            mlflow.log_metric("mse", float(mse))
            mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            return accuracy, mse, r2, mae