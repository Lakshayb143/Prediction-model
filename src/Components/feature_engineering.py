from src.logger import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


# This class defines a common interface for different feature engineering strategies.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame..

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass



# This strategy applies a logarithmic transformation to skewed features to normalize the distribution.
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation with the specific features to transform.
        """
        self.features = features

    def apply_transformation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = dataframe.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                dataframe[feature]
            )  # log1p handles log(0) by calculating log(1+x)
        logging.info("Log transformation completed.")
        return df_transformed



# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
  
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = dataframe.copy()
        df_transformed[self.features] = self.scaler.fit_transform(dataframe[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed



# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = dataframe.copy()
        df_transformed[self.features] = self.scaler.fit_transform(dataframe[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed



# This strategy applies one-hot encoding to categorical features, converting them into binary vectors.
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):

        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = dataframe.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(dataframe[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed


# Context Class for Feature Engineering

class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):

        logging.info(f"Switching feature engineering strategy from {self._strategy} to {strategy}.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
 
        logging.info(f"Applying feature engineering strategy with {self._strategy} strategy.")
        return self._strategy.apply_transformation(df)

