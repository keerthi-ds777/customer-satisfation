import numpy as np
import json
import pandas as pd 
from steps.ingest_data import ingesting
from steps.evaluate import evaluate_model
from steps.training_data import train_model
from steps.preprocessing import cleaning
from zenml import pipeline
from pydantic import BaseModel
from zenml.client import Client
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from src.deploy import MLFlowModelDeployerConfig
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.config import DockerSettings
import logging
from pipeline.utils import get_data_for_test
from src.data_cleaning import ModelTrainig
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
    model_name: str = "model",
    model_version: str = None,
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
        model_version: the version of the model to be deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        model_version=model_version,
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

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
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
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction



@pipeline(enable_cache=True, settings={'docker': docker_settings})
def continuous_deployment_pipeline(deploymentconfig: DeploymentTriggerConfig,
                                   mlflow_model_deployer_config: MLFlowModelDeployerConfig,
                                   model_name:str="rf_model",
                                   model_version:str) -> None:
    # Ingest data
    data = ingesting(data_path="data/olist_customers_dataset.csv")
    
    # Clean data
    #preprocessing the data
    x_train, x_test, y_train, y_test = cleaning(data)  
    
    # Train model
    

    model_training=ModelTrainig(x_train, x_test, y_train, y_test)
    model = model_training.fit_model(registerd_model=True,model_name=model_name,model_version=model_version) #train the model
    y_pred = model.predict(x_test)
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

@pipeline(enable_cache=True, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str, model_version: str = None):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
        model_version=model_version
    )
    predictor(service=model_deployment_service, data=batch_data)
