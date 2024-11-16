import pandas as pd
from src.Components.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling
)
from src.exception import CustomException
from src.logger import logging
import sys

from zenml import step

@step
def feature_engineering_stage(dataframe :pd.DataFrame, strategy :str = "StandardScaling", features :list = None) -> pd.DataFrame:

    if features is None:
        raise CustomException("No features were provided for feature engineering",sys)
    
    if strategy == "LogTransformation":
        featureEngineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "OneHotEncoding":
        featureEngineer = FeatureEngineer(OneHotEncoding(features))
    elif strategy == "StandardScaling":
        featureEngineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "MinMaxScaling":
        featureEngineer = FeatureEngineer(MinMaxScaling(features))

    else:
        raise CustomException(f"Unsupported feature engineering strategy: {strategy}",sys)
    

    logging.info(f"Starting feature engineering with strategy : {strategy}")
    

    transformed_df = featureEngineer.apply_feature_engineering(dataframe)

    logging.info(f"feature engineering completed successfully with strategy : {strategy}")
    
    return transformed_df
    