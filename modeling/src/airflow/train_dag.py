import os
import shutil
import sys

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(project_path)
sys.path.append(project_path)

from datetime import timedelta
from textwrap import dedent
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

from modeling.src.airflow.tasks.train import train

with DAG(
    "anomaly_model_train",
    default_args = {
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description="anomaly model train",
    schedule=timedelta(days=30),
    start_date=pendulum.now("UTC").subtract(days=1),
    catchup=False,
) as dag:
    train_task = PythonOperator(
        task_id='train_task',
        python_callable=train,
        op_kwargs={'project_path': project_path},
    )

    train_task

if __name__ == "__main__":
    train(project_path)