import io
import os
import glob
import sys

from dotenv import load_dotenv
load_dotenv()

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(project_path)
sys.path.append(project_path)

from datetime import timedelta
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

from modeling.src.utils.aws import download_data_from_s3
from modeling.src.train.train import run_temperature_train_on_airflow

data_root_path = os.path.join(project_path, 'data')

with DAG(
    "train_temperature_on_airflow",
    default_args = {
        'owner': 'lmw',
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description="train temperature on airflow",
    schedule=timedelta(days=30),
    start_date = pendulum.now("UTC").subtract(days=1),
    catchup=False,
) as dag:
    download_temperature_from_s3_task = PythonOperator(
        task_id='download_temperature_from_s3',
        python_callable=download_data_from_s3,
        op_kwargs={
            'data_root_path': data_root_path,
            'data_name': 'temperature',
        },
    )

    run_pm_train_task = PythonOperator(
        task_id='run_temperature_train_on_airflow',
        python_callable=run_temperature_train_on_airflow,
        op_kwargs={
            'data_root_path': data_root_path,
            'model_root_path': os.path.join(project_path, 'models'),
            'batch_size': 64,
            'model_name': 'multi_output_lstm'
        },
    )

    download_temperature_from_s3_task >> run_pm_train_task