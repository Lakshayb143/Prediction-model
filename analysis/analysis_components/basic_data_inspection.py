from abc import ABC, abstractmethod

from pandas import DataFrame


# It acts as a common interface for data inspection strategies.
class DataIngestionStrategy(ABC):

    @abstractmethod
    def inspect_data(self, dataframe : DataFrame) -> None:
        """
        Perform a specific type of data inspection and it prints the result directly
        """

# This strategy inspects the data types of each column and counts non-null values.
class DataTypesInspectionStrategy(DataIngestionStrategy):

    def inspect_data(self, dataframe : DataFrame) -> None:
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        """

        print("\n Data Types and Non-null Count:")
        print(dataframe.info())



# This strategy provides summary statistics for both numerical and categorical features.
class StatisticalSummaryInspectionStrategy(DataIngestionStrategy):

    def inspect_data(self, dataframe : DataFrame) -> None:

        """
        Prints summary statistics for numerical and categorical features.
        """

        print("\nSummary Statistics (Numerical Features):")
        print(dataframe.describe())

        print("\nSummary Statistics (Categorical Features):")
        print(dataframe.describe(include=["O"]))


class DataInspector:
    def __init__(self, strategy :DataIngestionStrategy):
        self.strategy = strategy

    def set_strategy(self, other_strategy :DataIngestionStrategy) -> None:
        self.strategy = other_strategy

    def data_inspection(self, dataframe :DataFrame) -> None:
        self.strategy.inspect_data(dataframe);