import mlflow

# -------------------------------
# 1. Connect to MLflow Server
# -------------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def register_best_model(run_id):
    model_uri = f"runs:/{run_id}/model"
    model_name = "housing-price-model"

    print(f"Registering model: {model_uri}")

    # Register the model in the MLflow Model Registry
    result = mlflow.register_model(model_uri, model_name)

    print(f"Model registered as version: {result.version}")

# -------------------------------
# 2. Input run ID at runtime
# -------------------------------
if __name__ == "__main__":
    # Note: In a real MLOps pipeline, this run_id might be passed via command-line arguments 
    # rather than user input.
    run_id = input("Enter MLflow run_id: ")
    register_best_model(run_id)