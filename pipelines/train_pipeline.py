
from stages.data_ingestion_stage import data_ingestion_stage
from stages.data_splitting_stage import data_splitting_stage
from stages.feature_engineering_stage import feature_engineering_stage
from stages.Handling_missing_values_stages import handle_missing_values_stage
from stages.model_building_stage import model_building_step
from zenml import Model, pipeline


@pipeline(
    model=Model(
        name="prices_predictor"
    ),
)


def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_stage(
        file_path="S:\\LB_Projects\\Prediction-model\\data\\archive.zip"
    )

    # Handling Missing Values Step
    filled_data = handle_missing_values_stage(raw_data)

    # Feature Engineering Step
    engineered_data = feature_engineering_stage(
        filled_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
    )


    # Data Splitting Step
    X_train, X_test, y_train, y_test = data_splitting_stage(engineered_data, target_column="SalePrice")

    # Model Building Step
    model = model_building_step(X_train=X_train, y_train=y_train)


    return model


if __name__ == "__main__":
    run = ml_pipeline()
