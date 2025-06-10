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
from textwrap import dedent
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

from modeling.src.train.anomaly import train
from modeling.src.utils.aws import download_data_from_s3, fast_download_data_from_s3

data_root_path = os.path.join(project_path, 'data')

with DAG(
    "train_anomaly_temperature_on_airflow",
    default_args = {
        'owner': 'lmw',
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description="train anomaly temperature on airflow",
    schedule=timedelta(days=30),
    start_date=pendulum.now("UTC").subtract(days=1),
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

    train_task = PythonOperator(
        task_id='train_task',
        python_callable=train,
        op_kwargs={
            'project_path': project_path,
            'data_name': 'temperature', 
            'model_name': 'temperature'
        },
    )

    download_temperature_from_s3_task >> train_task

if __name__ == "__main__":
    train(project_path)