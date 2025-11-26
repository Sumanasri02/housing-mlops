import os
import joblib
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Paths
MODEL_PATH = r"D:\Techflitter\housing-mlops\models\best_model.pkl"
TEST_DATA_PATH = r"D:\Techflitter\housing-mlops\data\preprocessed_data.csv"
METRICS_PATH = r"D:\Techflitter\housing-mlops\metrics\test_metrics.json"
PLOT_PATH = r"D:\Techflitter\housing-mlops\plots\prediction_vs_actual.png"

# Load model
model = joblib.load(MODEL_PATH)

# Load test data
df = pd.read_csv(TEST_DATA_PATH)
X_test = df.drop(columns=["price"])
y_test = df["price"]

# Predict
y_pred = model.predict(X_test)

# Compute metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

metrics = {"R2": r2, "MSE": mse, "MAE": mae}

# Save metrics
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation Metrics:", metrics)

# Plot predicted vs actual
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.savefig(PLOT_PATH)
plt.close()

# MLflow Logging
mlflow.set_experiment("Housing-Price-Evaluation")
with mlflow.start_run(run_name="best_model_evaluation"):
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(PLOT_PATH)
    mlflow.sklearn.log_model(model, "best_model")

print("Evaluation complete! Metrics and plot saved, logged to MLflow.")
