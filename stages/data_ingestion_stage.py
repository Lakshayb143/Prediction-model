import pandas as pd
from zenml import step

from src.logger import logging
from src.Components.data_ingestion import DataIngestorFactory





@step
def data_ingestion_stage(file_path :str) -> pd.DataFrame:
      
    """THis method will ingest data from a ZIP file using the appropriate DataIngestor."""
    
    logging.info(f"Starting data ingestion with file path {file_path}")
    data_ingestor  = DataIngestorFactory.get_data_ingestor(file_path)

    output_ingestor :dict = data_ingestor.ingest(file_path)

    logging.info(f"Data ingestion completed successfully with extracted file path {output_ingestor['file_path']}")

    dataframe : pd.DataFrame = output_ingestor['dataframe']

    return dataframe


