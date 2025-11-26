import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------
# Load Preprocessed Data
# ----------------------------------------------------------
data_path = r"D:\Techflitter\housing-mlops\data\preprocessed_data.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# Models to Train
# ----------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42)
}

# ----------------------------------------------------------
# Start MLflow Experiment
# ----------------------------------------------------------
mlflow.set_experiment("Housing-Price-Experiment")

results = {}
best_model = None
best_r2 = -999
best_model_name = ""


for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        preds = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Save in result dictionary
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

        # Log metrics to MLflow
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log model parameters
        mlflow.log_param("model_name", name)

        # Log the entire model into MLflow
        mlflow.sklearn.log_model(model, artifact_path=f"model_{name}")

        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name


# ----------------------------------------------------------
# Save Best Model Locally
# ----------------------------------------------------------
save_path = r"D:\Techflitter\housing-mlops\models\best_model.pkl"
joblib.dump(best_model, save_path)

print("\nBest Model:", best_model_name)
print("Saved at:", save_path)
print("All metrics logged to MLflow!")
