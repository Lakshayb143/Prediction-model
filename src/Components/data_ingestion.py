import os
import zipfile
from abc import ABC , abstractmethod
from src.exception import CustomException
from src.logger import logging
import pandas as pd

"""
We are using factory design pattern to perform Data Ingestion using different file types.
"""

# Defining an abstract class for Data Ingestion
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path : str) -> dict:
        """
        This method will perform Data Ingestion from a given file path.

        It will return a dictionary containing the following
        {
            "dataframe" : df,
            "file_path" : csv_file_path,
        }

        """
        pass


# Implementing a class for data ingestion using zip file.
class ZipDataIngestor(DataIngestor):

    def ingest(self, file_path :str) -> dict:

        if not file_path.endswith('.zip'):
            raise CustomException("The provided file is not a zip file.")

        with zipfile.ZipFile(file_path, "r") as zip_f:
            zip_f.extractall("artifacts/dataset")

        logging.info("Zip file extracted successfully.")

        extracted_files = os.listdir("artifacts/dataset")
        csv_data_files = [ file for file in extracted_files if file.endswith(".csv")]

        if len(csv_data_files) == 0:
            raise FileNotFoundError("There is no csv file")
        elif len(csv_data_files) > 1:
            raise ValueError("There are more than one csv file. Please choose one.")

        csv_file_path = os.path.join("artifacts/dataset", csv_data_files[0])
        df = pd.read_csv(csv_file_path)

        """
        Returning the DataFrame successfully
        """
        # logging.info("Returning the DataFrame successfully")

        return {
            "dataframe" : df,
            "file_path" : csv_file_path,
        }
    


class DataIngestorFactory:

    @staticmethod
    def get_data_ingestor(file_path :str) -> DataIngestor:

        file_extension = os.path.splitext(file_path)[1]

        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            logging.error(f"No data ingestor for file with extension {file_path}")
            raise ValueError(f"No data ingestor for file with extension {file_path} ")
            


if __name__ == "__main__":

    # file_path = "data/archive.zip"

    # data_ingestion = DataIngestorFactory.get_data_ingestor(file_path)

    # dataFrame = data_ingestion.ingest(file_path)

    # print(dataFrame["dataframe"].head())

    pass

 
