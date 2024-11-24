from abc import ABC, abstractmethod

from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


"""
Here we are using Template design pattern to create the templates for Missing Value Analysis.
"""


# This class defines a template for missing values analysis.
class MissingValueAnalysisTemplate(ABC):

    @abstractmethod
    def identify_missing_values(self, dataframe :DataFrame):
        """
        This method should print the count of missing values for each column.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, dataframe :DataFrame):
        """
        Visualizes missing values in the dataframe. THis method will create a visualization
        """
        pass


    def analyze(self, dataframe :DataFrame) -> None:
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.
        """
        self.identify_missing_values(dataframe)
        self.visualize_missing_values(dataframe)



# This class implements methods to identify and visualize missing values in the dataframe.
class MissingValueAnalysis(MissingValueAnalysisTemplate):
    def identify_missing_values(self, dataframe: DataFrame) -> None:
        print("\n Missing Values Count by Column:")
        missing_values = dataframe.isnull().sum()
        print(missing_values[missing_values > 0])


    def visualize_missing_values(self, dataframe: DataFrame) -> None:
        print("\nVisualizing Missing Values...")
        plt.figure(figsize=(12, 8))
        sns.heatmap(dataframe.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
