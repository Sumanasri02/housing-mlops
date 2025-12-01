import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

with mlflow.start_run():
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("model_type", "YourModelNameHere")  # update with your model

    # your training code...
    # mlflow.log_metric("mse", mse_value)
    # mlflow.log_metric("rmse", rmse_value)
