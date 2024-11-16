from src.logger import logging
from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

# This class defines a common interface for different data splitting strategies.
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, dataframe: pd.DataFrame, target_column: str):
        """
        Abstract method to split the data into training and testing sets

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        pass


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, dataframe: pd.DataFrame, target_column: str):
        """
        Splits the data into training and testing sets using a simple train-test split..

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Performing simple train-test split.")
        X = dataframe.drop(columns=[target_column])
        y = dataframe[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logging.info("Train-test split completed.")
        return X_train, X_test, y_train, y_test


# Context Class for Data Splitting
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):

        logging.info(f"Switching data splitting strategy to {strategy}.")
        self._strategy = strategy

    def split(self, dataframe: pd.DataFrame, target_column: str):
        """
        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info(f"Splitting data using the selected ({self._strategy}) strategy on target column {target_column}.")
        return self._strategy.split_data(dataframe, target_column)



if __name__ == "__main__":
    
    # df = pd.read_csv('your_data.csv')

    # Initialize data splitter with a specific strategy
    # data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    # X_train, X_test, y_train, y_test = data_splitter.split(df, target_column='SalePrice')

    pass
