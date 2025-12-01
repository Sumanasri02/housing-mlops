import pandas as pd
import joblib
import os
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

INPUT_PATH = "data/preprocessed_data.csv"
MODEL_PATH = "models/best_model.pkl"

def main():
    try:
        df = pd.read_csv(INPUT_PATH)
        X = df.drop(columns=["price"])
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mlflow.set_experiment("Housing-Model-Training")

        models = {
            "RandomForest": RandomForestRegressor(random_state=42),
            "GradientBoost": GradientBoostingRegressor(random_state=42)
        }

        best_model = None
        best_r2 = -999

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)

                mlflow.log_metric("R2", r2)

                if r2 > best_r2:
                    best_model = model
                    best_r2 = r2

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        print("🧠 Best model saved:", MODEL_PATH)
        return True

    except Exception as e:
        print("❌ Model training failed:", e)
        return False
