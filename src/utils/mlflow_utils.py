# src/utils/mlflow_utils.py
# centralizes all mlflow setup and logging helpers

import os
import subprocess
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from src.utils.logger import get_logger

uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/da25s003/E2E_Project_DA5402.mlflow")

load_dotenv()
logger = get_logger("mlflow_utils")

def setup_mlflow(experiment_name: str) -> str:
    """Configure MLflow to log to DagsHub and return active run id."""
    mlflow.set_tracking_uri(uri=uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")
    logger.info(f"Experiment: {experiment_name}")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).strip().decode("utf-8")
    except Exception:
        return "unknown"


def get_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).strip().decode("utf-8")
    except Exception:
        return "unknown"


def log_tags(model_name: str, stage: str):
    """Log git and run metadata as MLflow tags."""
    mlflow.set_tags({
        "git_commit" : get_git_commit(),
        "git_branch" : get_git_branch(),
        "model_name" : model_name,
        "stage"      : stage,
        "run_by"     : "dvc" if os.environ.get("DVC_ROOT") else "manual",
    })
    logger.info(f"Git commit: {get_git_commit()}")


def log_params_from_dict(params: dict):
    """Log all params — MLflow has 500 char limit per value."""
    mlflow.log_params({
        k: str(v)[:500] for k, v in params.items()
    })


def log_per_class_metrics(metrics_dict: dict, prefix: str, step: int = None):
    """
    Log per class metrics with consistent naming.
    e.g. prefix='val_f1' → val_f1_mel, val_f1_nv ...
    """
    for cls, value in metrics_dict.items():
        key = f"{prefix}_{cls}"
        if step is not None:
            mlflow.log_metric(key, value, step=step)
        else:
            mlflow.log_metric(key, value)


def delete_mlflow_registry():
    mlflow.set_tracking_uri(uri=uri)

    client = MlflowClient()

    # delete all versions first — required before deleting model
    versions = client.search_model_versions("name='skin-disease-classifier'")
    for v in versions:
        client.delete_model_version(
            name="skin-disease-classifier",
            version=v.version
        )
        print(f"Deleted version {v.version}")

    # then delete the registered model itself
    client.delete_registered_model(name="skin-disease-classifier")
    print("Model registry deleted")