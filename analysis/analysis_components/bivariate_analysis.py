from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

# This class defines a common interface for bivariate analysis strategies.
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, dataframe :DataFrame, feature_1 :str , feature_2 :str):
        """
        Perform bivariate analysis on two features of the dataframe.
        """
        pass


# This strategy analyzes the relationship between two numerical features using scatter plots.
class NumericalAndNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, dataframe: DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features using a scatter plot.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=dataframe)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


class CategoricalAndNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between a categorical feature and a numerical feature using a box plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature/column to be analyzed.
        feature2 (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a box plot showing the relationship between the categorical and numerical features.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()



class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):

        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: DataFrame, feature1: str, feature2: str):
        self._strategy.analyze(df, feature1, feature2)