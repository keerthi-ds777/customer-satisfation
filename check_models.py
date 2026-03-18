import mlflow
from mlflow.tracking import MlflowClient

try:
    client = MlflowClient()
    models = client.search_registered_models()
    print(f"Found {len(models)} registered models.")
    for rm in models:
        print(f"Name: {rm.name}")
        for v in rm.latest_versions:
            print(f"  Version: {v.version}, Stage: {v.current_stage}")
except Exception as e:
    print(f"Error listing models: {e}")
