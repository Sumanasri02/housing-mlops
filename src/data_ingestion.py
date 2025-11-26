import pandas as pd
import os  
import logging

logging.basicConfig(
   filename=r"D:\Techflitter\housing-mlops\logs\data_ingestion.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

class DataIngestion:
    def __init__(self, data_path: str):
        self.data_path = data_path
    def load_data(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.data_path):
                logging.error(f"File not found: {self.data_path}")
                raise FileNotFoundError(f"File not found: {self.data_path}")
            
            df = pd.read_csv(self.data_path)
            logging.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logging.exception("Error while loading data")
            raise e 
if __name__ == "__main__":
    ingestion = DataIngestion(data_path=r"D:\Techflitter\housing-mlops\data\Housing.csv")
    df = ingestion.load_data()
    print(df.head())