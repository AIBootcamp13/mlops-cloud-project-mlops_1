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
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

from modeling.src.airflow.tasks.download_data_from_s3 import download_pm10_from_s3
from modeling.src.train.train import run_pm_train

data_root_path = os.path.join(project_path, 'data')

def pm10_data_exists():
    files = glob.glob(os.path.join(project_path, "*_PM10_data.csv"))
    if files:
        return True
    return False

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
    start_data = pendulum.now("UTC").subtract(days=1),
    catchup=False,
) as dag:
    pm10_data_exists_check_task = BranchPythonOperator(
        task_id="pm10_data_exists_check",
        python_callable=pm10_data_exists
    )

    download_pm10_from_s3_task = PythonOperator(
        task_id='download_pm10_from_s3',
        python_callable=download_pm10_from_s3
    )

    run_pm_train_task = PythonOperator(
        task_id='run_pm_train',
        python_callable=run_pm_train,
        # ata_root_path, model_root_path, batch_size, model_name="MULTI_OUTPUT_LSTM"
        op_kwargs={
            'data_root_path': data_root_path,
            'model_root_path': os.path.join(project_path, 'models'),
            'batch_size': 64,
            'model_name': 'multi_output_lstm'
        },
    )

    end = EmptyOperator(task_id='pm10 train end')

    pm10_data_exists_check_task >> run_pm_train_task
    pm10_data_exists_check_task >> download_pm10_from_s3_task >> run_pm_train_task
    run_pm_train_task >> end
    
if __name__ == "__main__":
    download_pm10_from_s3(data_root_path)