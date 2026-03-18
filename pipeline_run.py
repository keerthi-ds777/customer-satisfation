import logging
from zenml import pipeline
from zenml.client import Client
from steps.preprocessing import cleaning
from steps.evaluate import evaluate_model
from pipeline._run_pipeline import run_pipeline

if __name__ =="__main__":
 print(Client().active_stack.experiment_tracker.get_tracking_uri())
 run_pipeline(data_path= r"data\olist_customers_dataset.csv")
 print(
        "Now run \n "
        f"mlflow ui --backend-store-uri '{Client().active_stack.experiment_tracker.get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )