import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

class MLflowTracker:
    """
    Production-ready MLflow tracking system for housing price prediction
    """
    
    def __init__(self, tracking_uri="http://127.0.0.1:5000", experiment_name="housing_price_prediction"):
        """
        Initialize MLflow tracker
        
        Args:
            tracking_uri: MLflow server URI
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
        
        mlflow.set_experiment(self.experiment_name)
        print(f"✓ MLflow connected to: {self.tracking_uri}")
        print(f"✓ Experiment: {self.experiment_name} (ID: {self.experiment_id})")
    
    def log_training_run(self, model, model_name, X_train, X_test, y_train, y_test, 
                        preprocessor=None, params=None, tags=None):
        """
        Log a complete training run with model, metrics, and artifacts
        
        Args:
            model: Trained sklearn model
            model_name: Name of the model (e.g., 'LinearRegression', 'RandomForest')
            X_train, X_test, y_train, y_test: Train/test data
            preprocessor: Preprocessing pipeline (optional)
            params: Dictionary of model parameters
            tags: Dictionary of tags for the run
        
        Returns:
            run_id: MLflow run ID
        """
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred, prefix="train")
            test_metrics = self._calculate_metrics(y_test, y_test_pred, prefix="test")
            
            # Log parameters
            if params:
                mlflow.log_params(params)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            # Log metrics
            mlflow.log_metrics(train_metrics)
            mlflow.log_metrics(test_metrics)
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            mlflow.set_tag("model_type", model_name)
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=None  # Don't auto-register
            )
            
            # Log preprocessor if provided
            if preprocessor:
                mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessor")
            
            # Create and log prediction comparison
            self._log_prediction_comparison(y_test, y_test_pred)
            
            run_id = run.info.run_id
            
            print(f"\n{'='*60}")
            print(f"✓ Run completed successfully!")
            print(f"{'='*60}")
            print(f"Run ID: {run_id}")
            print(f"Model: {model_name}")
            print(f"\nTest Metrics:")
            print(f"  - MAE:  {test_metrics['test_MAE']:.2f}")
            print(f"  - RMSE: {test_metrics['test_RMSE']:.2f}")
            print(f"  - R²:   {test_metrics['test_R2']:.4f}")
            print(f"{'='*60}\n")
            
            return run_id
    
    def _calculate_metrics(self, y_true, y_pred, prefix=""):
        """Calculate regression metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        prefix = f"{prefix}_" if prefix else ""
        
        return {
            f"{prefix}MAE": mae,
            f"{prefix}MSE": mse,
            f"{prefix}RMSE": rmse,
            f"{prefix}R2": r2
        }
    
    def _log_prediction_comparison(self, y_true, y_pred):
        """Log prediction vs actual comparison as CSV"""
        comparison_df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'error': y_true - y_pred,
            'abs_error': np.abs(y_true - y_pred)
        })
        
        # Save to temp file and log
        temp_file = "temp_predictions.csv"
        comparison_df.to_csv(temp_file, index=False)
        mlflow.log_artifact(temp_file, artifact_path="predictions")
        os.remove(temp_file)
    
    def compare_models(self, experiment_name=None):
        """
        Compare all models in the experiment
        
        Returns:
            DataFrame with all runs and their metrics
        """
        if experiment_name is None:
            experiment_name = self.experiment_name
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        # Select relevant columns
        columns = ['run_id', 'start_time', 'params.model_name', 
                  'metrics.test_MAE', 'metrics.test_RMSE', 'metrics.test_R2']
        
        available_columns = [col for col in columns if col in runs.columns]
        comparison_df = runs[available_columns].sort_values('metrics.test_R2', ascending=False)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80 + "\n")
        
        return comparison_df
    
    def get_best_model(self, metric='test_R2', mode='max'):
        """
        Get the best model based on a metric
        
        Args:
            metric: Metric to optimize (default: test_R2)
            mode: 'max' or 'min' (default: max)
        
        Returns:
            best_run_id, best_metric_value
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        metric_col = f'metrics.{metric}'
        
        if metric_col not in runs.columns:
            raise ValueError(f"Metric '{metric}' not found in runs")
        
        if mode == 'max':
            best_run = runs.loc[runs[metric_col].idxmax()]
        else:
            best_run = runs.loc[runs[metric_col].idxmin()]
        
        best_run_id = best_run['run_id']
        best_metric_value = best_run[metric_col]
        
        print(f"\n✓ Best model found!")
        print(f"  Run ID: {best_run_id}")
        print(f"  {metric}: {best_metric_value:.4f}")
        print(f"  Model: {best_run.get('params.model_name', 'Unknown')}")
        
        return best_run_id, best_metric_value
    
    def register_model(self, run_id, model_name="housing-price-model", stage="Staging"):
        """
        Register a model to MLflow Model Registry
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            stage: Stage to transition to (None, Staging, Production, Archived)
        
        Returns:
            model_version
        """
        model_uri = f"runs:/{run_id}/model"
        
        print(f"\nRegistering model...")
        print(f"  Run ID: {run_id}")
        print(f"  Model name: {model_name}")
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        print(f"✓ Model registered as version: {model_version.version}")
        
        # Transition to stage if specified
        if stage:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            print(f"✓ Model transitioned to: {stage}")
        
        return model_version
    
    def load_model_from_registry(self, model_name="housing-price-model", stage="Production"):
        """
        Load a model from the registry
        
        Args:
            model_name: Registered model name
            stage: Model stage (Production, Staging, etc.)
        
        Returns:
            Loaded model
        """
        model_uri = f"models:/{model_name}/{stage}"
        
        print(f"\nLoading model from registry...")
        print(f"  Model: {model_name}")
        print(f"  Stage: {stage}")
        
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✓ Model loaded successfully!")
        
        return model


# Example usage functions
def track_single_model():
    """Example: Track a single model"""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    # Initialize tracker
    tracker = MLflowTracker()
    
    # Load data
    df = pd.read_csv('data/processed/train.csv')
    X = df.drop('price', axis=1)
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Track with MLflow
    run_id = tracker.log_training_run(
        model=model,
        model_name="LinearRegression",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        params={'solver': 'auto'},
        tags={'version': 'v1.0', 'data': 'housing'}
    )
    
    return run_id


def track_multiple_models():
    """Example: Track multiple models and find the best"""
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    # Initialize tracker
    tracker = MLflowTracker()
    
    # Load data
    df = pd.read_csv('data/processed/train.csv')
    X = df.drop('price', axis=1)
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models to train
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and track all models
    run_ids = []
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")
        
        model.fit(X_train, y_train)
        
        run_id = tracker.log_training_run(
            model=model,
            model_name=name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=model.get_params()
        )
        run_ids.append(run_id)
    
    # Compare all models
    tracker.compare_models()
    
    # Get best model
    best_run_id, best_score = tracker.get_best_model(metric='test_R2')
    
    # Register best model
    tracker.register_model(best_run_id, stage="Production")
    
    return best_run_id


if __name__ == "__main__":
    # Example: Track a single model
    print("Starting MLflow tracking...")
    run_id = track_single_model()
    print(f"\nCompleted! Run ID: {run_id}")