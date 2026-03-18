import logging
from src.data_cleaning import DataCleaning
import pandas as pd
from steps.preprocessing import cleaning
from steps.evaluate import evaluate_model
from steps.ingest_data import ingesting
from steps.training_data import train_model
from zenml import step
from pydantic import BaseModel
from zenml.integrations.mlflow.services import MLFlowDeploymentService

class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.5


def get_data_for_test():
    try:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        df = df[[
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "review_score"
    ]]
        df = df.sample(n=100)
        x,y = DataCleaning(df).preprocess()
        result = df.to_json(orient="split")
        
        return result
    except Exception as e:
        logging.error(e)

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data


@step
def deployment_trigger_config(accuracy: float,
                              config: DeploymentTriggerConfig
) -> bool:
    """
    Step to configure deployment trigger settings.

    Args:
        deployment_trigger_config (DeploymentTriggerConfig): Configuration for deployment triggers.

    Returns:
        DeploymentTriggerConfig: The same configuration passed in.


    """
    return config.min_accuracy <= accuracy

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

class MlflowDeploymentLoaderStepParameter(BaseModel):
    pipeline_name: str
    step_name: str
    running: str = True


@step(enable_cache=True)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "rf_model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]
