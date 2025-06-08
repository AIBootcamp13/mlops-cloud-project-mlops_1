from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from src.data_loaders.temp_loader import main as load_temp
from src.data_loaders.pm10_loader import main as load_pm10
from src.data_processors.eda_processor import run_eda_for_recent_days
from src.cloud.s3_uploader import upload_to_s3
from datapipeline.airflow.utils.slack_notifier import notify_slack

def run_eda_and_upload(execution_date, **kwargs):
    reference_date = pd.to_datetime(execution_date)
    temp_dates, pm10_dates = run_eda_for_recent_days(days=14, reference_date=reference_date)
    upload_to_s3(temp_dates, pm10_dates)

default_args = {
    'owner': 'eunbyul',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': notify_slack,
    'on_success_callback': notify_slack,
}

with DAG(
    dag_id='weather_pipeline',
    default_args=default_args,
    schedule_interval='0 4 * * *',
    catchup=False,
    tags=['mlops'],
) as dag:

    task_temp = PythonOperator(
        task_id='load_temperature_data',
        python_callable=load_temp,
    )

    task_pm10 = PythonOperator(
        task_id='load_pm10_data',
        python_callable=load_pm10,
    )

    task_eda_upload = PythonOperator(
        task_id='run_eda_and_upload',
        python_callable=run_eda_and_upload,
        op_kwargs={'execution_date': '{{ ds }}'},
    )

    task_temp >> task_pm10 >> task_eda_upload