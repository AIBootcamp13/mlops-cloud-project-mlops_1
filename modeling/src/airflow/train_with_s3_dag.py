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

from modeling.src.utils.utils import download_pm10_from_s3
from modeling.src.train.train import run_pm_train_with_s3

data_root_path = os.path.join(project_path, 'data')

with DAG(
    "train_pm10_with_s3",
    default_args = {
        'owner': 'lmw',
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description="train with s3",
    schedule=timedelta(days=30),
    start_date = pendulum.now("UTC").subtract(days=1),
    catchup=False,
) as dag:

    download_pm10_from_s3_task = PythonOperator(
        task_id='download_pm10_from_s3',
        python_callable=download_pm10_from_s3,
        op_kwargs={
            'data_root_path': data_root_path,
        },
    )

    run_pm_train_task = PythonOperator(
        task_id='run_pm_train_with_s3',
        python_callable=run_pm_train_with_s3,
        op_kwargs={
            'data_root_path': data_root_path,
            'model_root_path': os.path.join(project_path, 'models'),
            'batch_size': 64,
            'model_name': 'multi_output_lstm'
        },
    )

    download_pm10_from_s3_task >> run_pm_train_task
    
if __name__ == "__main__":
    download_pm10_from_s3(data_root_path)