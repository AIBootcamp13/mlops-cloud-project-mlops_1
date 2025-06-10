from datetime import datetime, timedelta
from pendulum import timezone
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd

from src.data_loaders.temp_loader import run_temp_preprocessing as load_temp
from src.data_loaders.pm10_loader import run_pm10_preprocessing as load_pm10
from src.data_processors.eda_processor import run_eda_for_recent_days_with_fetch
from src.cloud.s3_uploader import upload_processed_data_to_s3

from datapipeline.airflow.utils.slack_notifier import notify_slack

# 실행 기준 날짜에서 하루 전을 기준으로 EDA 진행
def run_eda_and_upload(**context):
    reference_date = pd.to_datetime('today') - pd.Timedelta(days=1)
    df_temp, df_pm10 = run_eda_for_recent_days_with_fetch(days=14, reference_date=reference_date)
    upload_processed_data_to_s3(df_temp, df_pm10)

default_args = {
    'owner': 'eunbyul',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1, tzinfo=timezone("UTC")),  # 과거 날짜로 설정
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': notify_slack,
    'on_success_callback': notify_slack,
}

with DAG(
    dag_id='weather_pipeline',
    default_args=default_args,
    schedule_interval='0 19 * * *',  # KST 04:00 = UTC 19:00
    catchup=False,
    tags=['mlops'],
    max_active_runs=1,  # 동시 실행 제한 추가
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
        provide_context=True,
    )

    task_temp >> task_pm10 >> task_eda_upload