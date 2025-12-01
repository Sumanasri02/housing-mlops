import pandas as pd
import os
import logging

LOG_PATH = "logs/data_ingestion.log"
DATA_INPUT_PATH = "data/Housing.csv"
DATA_OUTPUT_PATH = "data/ingested_data.csv"

# Make sure logs folder exists
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

def main():
    try:
        if not os.path.exists(DATA_INPUT_PATH):
            logging.error(f"File not found: {DATA_INPUT_PATH}")
            print("❌ Data ingestion failed: File missing")
            return False

        df = pd.read_csv(DATA_INPUT_PATH)

        # Ensure output folder exists
        os.makedirs(os.path.dirname(DATA_OUTPUT_PATH), exist_ok=True)

        df.to_csv(DATA_OUTPUT_PATH, index=False)

        logging.info(f"Data ingested successfully → {df.shape}")
        print(f"📥 Data ingested successfully → {df.shape}")
        return True

    except Exception as e:
        logging.exception("Error in Data Ingestion")
        print("❌ Data ingestion error:", e)
        return False
