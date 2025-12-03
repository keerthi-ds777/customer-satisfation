from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentConfig, MLFlowDeploymentService

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import mlflow.sklearn
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT

from pydantic import BaseModel

from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.model_deployers.base_model_deployer import BaseModelDeployer
import os
class MLFlowModelDeployerConfig(BaseModel):
    workers: int = 1
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT



