import mlflow
import mlflow.sklearn
import pandas as pd

# --------------------------------------
# Correct path to your preprocessed data
# --------------------------------------
data_path = r"D:\Techflitter\housing-mlops\data\preprocessed_data.csv"

# Load the preprocessed dataset
df = pd.read_csv(data_path)

# --------------------------------------
# Create MLflow Experiment
# --------------------------------------
mlflow.set_experiment("Housing-Price-Experiment")

with mlflow.start_run():

    # Log the preprocessed CSV file
    mlflow.log_artifact(data_path, artifact_path="preprocessed_data")

    print("✔ preprocessed_data.csv logged successfully to MLflow")

print("\nMLflow setup step completed!")
