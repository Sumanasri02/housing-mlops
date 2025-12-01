from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=2),
    "start_date": datetime(2025, 1, 1),
}

with DAG(
    dag_id="train_pipeline_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "housing"],
) as dag:

    # 1️⃣ Data Ingestion
    run_ingestion = BashOperator(
        task_id="run_data_ingestion",
        bash_command=(
            "python /opt/airflow/src/data_ingestion.py "
            "1>> /opt/airflow/logs/data_ingestion.log 2>> /opt/airflow/logs/data_ingestion.err"
        ),
    )

    # 2️⃣ Data Preprocessing
    run_preprocess = BashOperator(
        task_id="run_preprocessing",
        bash_command=(
            "python /opt/airflow/src/preprocessing/data_preprocessing.py "
            "1>> /opt/airflow/logs/preprocess.log 2>> /opt/airflow/logs/preprocess.err"
        ),
    )

    # 3️⃣ Exploratory Data Analysis (EDA)
    run_eda = BashOperator(
        task_id="run_eda",
        bash_command=(
            "python /opt/airflow/src/eda/eda.py "
            "1>> /opt/airflow/logs/eda.log 2>> /opt/airflow/logs/eda.err"
        ),
    )

    # 4️⃣ Model Training
    run_training = BashOperator(
        task_id="run_training",
        bash_command=(
            "python /opt/airflow/src/model_trainer/model_trainer.py "
            "1>> /opt/airflow/logs/training.log 2>> /opt/airflow/logs/training.err"
        ),
    )

    # 5️⃣ Model Evaluation
    run_evaluation = BashOperator(
        task_id="run_evaluation",
        bash_command=(
            "python /opt/airflow/src/evaluation/evaluate_model.py "
            "1>> /opt/airflow/logs/eval.log 2>> /opt/airflow/logs/eval.err"
        ),
    )

    # 6️⃣ MLflow Model Register
    mlflow_register = BashOperator(
        task_id="mlflow_register",
        bash_command=(
            "python /opt/airflow/src/mlflow_pipeline/mlflow_setup.py "
            "1>> /opt/airflow/logs/mlflow_register.log 2>> /opt/airflow/logs/mlflow_register.err || true"
        ),
    )

    # Ordering
    run_ingestion >> run_preprocess >> run_eda >> run_training >> run_evaluation >> mlflow_register
