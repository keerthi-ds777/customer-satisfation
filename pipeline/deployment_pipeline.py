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
from pipeline.utils import deployment_trigger_config, dynamic_importer, prediction_service_loader
from pipeline.utils import MlflowDeploymentLoaderStepParameter, DeploymentTriggerConfig
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
docker_settings = DockerSettings(required_integrations=[MLFLOW])




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
def continuous_deployment_pipeline(
    min_accuracy: float = 0.5,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
    model_name:str="rf_model",
    model_version:str="1"
) -> None:
    
    deploymentconfig = DeploymentTriggerConfig(min_accuracy=min_accuracy)
    mlflow_model_deployer_config = MLFlowModelDeployerConfig(workers=workers, timeout=timeout)

    data = ingesting(data_path="data/olist_customers_dataset.csv")
    
    # Clean data
    #preprocessing the data
    x_train, x_test, y_train, y_test = cleaning(data)  
    model, y_pred = train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        registerd_model=True,
        model_name=model_name,
        model_version=model_version
    )
    
    
    # Evaluate model
    accuracy, mse, r2, mae = evaluate_model(y_test, y_pred)
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
