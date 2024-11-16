import pandas as pd
from zenml import step

from src.Components.data_ingestion import DataIngestorFactory





@step
def data_ingestion_step(file_path :str) -> pd.DataFrame:
      
    """THis method will ingest data from a ZIP file using the appropriate DataIngestor."""

    data_ingestor  = DataIngestorFactory.get_data_ingestor(file_path)

    output_ingestor :dict = data_ingestor.ingest(file_path)

    dataframe : pd.DataFrame = output_ingestor['dataframe']

    return dataframe


