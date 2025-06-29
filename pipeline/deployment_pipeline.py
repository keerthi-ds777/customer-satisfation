import numpy 
import pandas as pd 
from steps.ingest_data import ingesting
from steps.evaluate import evaluate_model
from steps.training_data import train_model
from steps.preprocessing import cleaning
from zenml import pipeline
from pydantic import BaseModel
from zenml.client import Client
from zenml import step
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from src.deploy import MLFlowModelDeployerConfig
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.config import DockerSettings
import logging
docker_settings = DockerSettings(required_integrations=[MLFLOW])
class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.5

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


@pipeline(enable_cache=False, settings={'docker': docker_settings})
def continuous_deployment_pipeline(deploymentconfig: DeploymentTriggerConfig,
                                   mlflow_model_deployer_config: MLFlowModelDeployerConfig) -> None:
    # Ingest data
    data = ingesting(data_path="data/olist_customers_dataset.csv")
    
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
    
    deployment_decision = deployment_trigger_config(accuracy=accuracy, config=deploymentconfig)

    mlflow_model_deployer_step(model=model,
    workers=mlflow_model_deployer_config.workers,
    timeout=mlflow_model_deployer_config.timeout,
    deploy_decision=deployment_decision

    )
