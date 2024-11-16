from typing import Tuple
import pandas as pd
from src.Components.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy
from src.logger import logging

from zenml import step


@step
def data_splitting_stage(dataframe: pd.DataFrame, target_column :str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy."""

    splitter = DataSplitter(SimpleTrainTestSplitStrategy())
    logging.info("Starting data splitting stage...")
    X_train, X_test, y_train, y_test = splitter.split(dataframe,target_column)

    logging.info("Data splitting completed successfully.")
    return X_train, X_test, y_train, y_test
