from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentConfig, MLFlowDeploymentService

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import mlflow.sklearn
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT

from zenml.models.v2.misc.service import ServiceType
from pydantic import BaseModel

class MLFlowModelDeployerConfig(BaseModel):
    
    """Configuration for the MLFlow model deployer step."""


    
    workers:int=3
    timeout:int= 60
    data_path: str = r"data\olist_customers_dataset.csv"
    
