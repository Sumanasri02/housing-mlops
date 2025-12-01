import mlflow
import mlflow.sklearn
import pandas as pd
import os

DATA_PATH = "/opt/airflow/data/preprocessed_data.csv"
os.makedirs("/opt/airflow/logs", exist_ok=True)

df = pd.read_csv(DATA_PATH)

mlflow.set_experiment("Housing-Price-Experiment")
with mlflow.start_run():
    mlflow.log_artifact(DATA_PATH, artifact_path="preprocessed_data")
    print("✔ preprocessed_data.csv logged successfully to MLflow")

print("\nMLflow setup step completed!")
