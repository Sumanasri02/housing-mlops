import joblib
import pandas as pd
import json
import os
from sklearn.metrics import r2_score, mean_squared_error

MODEL_PATH = "models/best_model.pkl"
TEST_DATA_PATH = "data/preprocessed_data.csv"
METRICS_PATH = "metrics/test_metrics.json"

def main():
    try:
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(TEST_DATA_PATH)
        X = df.drop(columns=["price"])
        y = df["price"]

        preds = model.predict(X)

        metrics = {
            "R2": r2_score(y, preds),
            "RMSE": mean_squared_error(y, preds, squared=False),
        }

        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        print("📊 Model evaluation complete")
        return True

    except Exception as e:
        print("❌ Evaluation failed:", e)
        return False
