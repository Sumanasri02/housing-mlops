FROM apache/airflow:2.10.3

USER airflow
RUN pip install --no-cache-dir \
    "mlflow[extras]" \
    pandas \
    scikit-learn \
    psycopg2-binary

    