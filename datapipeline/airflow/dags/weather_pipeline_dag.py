from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.data_loaders.temp_loader import main as load_temp
from src.data_loaders.pm10_loader import main as load_pm10
from src.data_processors.eda_processor import main as run_eda
from src.cloud.s3_uploader import main as upload_s3

default_args = {
    'owner': 'eunbyul',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='weather_pipeline',
    default_args=default_args,
    schedule_interval='0 4 * * *',  # 매일 04:00에 실행
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

    task_eda = PythonOperator(
        task_id='run_eda',
        python_callable=run_eda,
    )

    task_upload = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_s3,
    )

    task_temp >> task_pm10 >> task_eda >> task_upload