
from pipelines.train_pipeline import ml_pipeline
from stages.data_importer import data_importer
from stages.prediction_service_loader import prediction_service_loader
from stages.predictor import predictor

from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step


@pipeline
def continuous_deployment_pipeline():
    """Run a training job and deploy an MLflow model deployment."""

    trained_model = ml_pipeline()

    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)



@pipeline
def inference_pipeline():
    """Run a batch inference job with data loaded from an API."""

    batch_data = data_importer()

    model_deployment_service = prediction_service_loader(
        pipline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step"
    )

    predictor(service=model_deployment_service, input_data=batch_data)
