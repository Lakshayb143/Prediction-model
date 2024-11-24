from src.exception import CustomException
import logging
from typing import Annotated
import sys


import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name="prices_predictor",
    version="1.0.0",
    license="MIT",
    description="Price prediction model for houses.",
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> Annotated[Pipeline, ArtifactConfig(name="trained_pipeline")]:
    """
    Builds and trains a Linear Regression model using scikit-learn wrapped in a pipeline.

    Returns:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the Linear Regression model.
    """
#   Validating the inputs datasets
    if not isinstance(X_train, pd.DataFrame):
        raise CustomException("X_train must be a pandas DataFrame.",sys)
    if not isinstance(y_train, pd.Series):
        raise CustomException("y_train must be a pandas Series.",sys)

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # Define preprocessing for categorical and numerical features
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Define the model training pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])

    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()  # Start a new MLflow run if there isn't one active

    try:
        # Enable autologging for scikit-learn to automatically capture model metrics, parameters, and artifacts
        mlflow.sklearn.autolog()

        logging.info("Building and training the Linear Regression model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Log the columns that the model expects
        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_cols])
        expected_columns = numerical_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )
        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()

    return pipeline
