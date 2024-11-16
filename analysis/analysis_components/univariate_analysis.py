from abc import ABC, abstractmethod

from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

class UnivariateAnalysisStrategy(ABC):

    @abstractmethod
    def analyze(self, dataframe :DataFrame, feature :str) -> None:
        """
        Perform univariate analysis on a specific feature of the dataframe.
        """
        pass


# This strategy analyzes numerical features by plotting their distribution.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, dataframe: DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature using a histogram and KDE.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(dataframe[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()



# This strategy analyzes categorical features by plotting their frequency distribution.
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: DataFrame, feature: str):
        """
        Plots the distribution of a categorical feature using a bar plot.
        Displays a bar plot showing the frequency of each category.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()



class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy) -> None:
        self._strategy = strategy

    def execute_analysis(self, df: DataFrame, feature: str) -> None:
        """
        Executes the univariate analysis using the current strategy.
        """
        self._strategy.analyze(df, feature)