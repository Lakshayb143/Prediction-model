import pandas as pd
from src.Components.handling_missing_values import (
    MissingValueHandler,
    DropMissingValuesStrategy,
    FillMissingValuesStrategy
)


from src.exception import CustomException
from src.logger import logging
import sys

from zenml import step

@step
def handle_missing_values_stage(dataframe :pd.DataFrame, strategy :str = "mean") -> pd.DataFrame:

    """Handles missing values using MissingValueHandler and the specified strategy."""
    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStrategy(axis=0, thresh=5))

    elif strategy == ["mean", "median", "constant", "mode"]:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))

    else:
        raise CustomException(f"Unsupported missing value handling strategy: {strategy}",sys)
    
    logging.info(f"Handling missing values using {strategy} strategy")

    cleaned_df = handler.handle_missing_values(dataframe)
    return cleaned_df




