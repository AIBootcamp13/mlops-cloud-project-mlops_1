from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

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

    task_upload_s3 = DockerOperator(
        task_id='upload_to_s3_docker',
        image='upload-script:latest',
        api_version='auto',
        auto_remove=True,
        command='sh -c "PYTHONPATH=/app python src/cloud/upload_script.py --temp_dates=2025-06-01,2025-06-02 --pm10_dates=2025-06-01,2025-06-02"',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        environment={
            'AWS_ACCESS_KEY_ID': 'your_key',
            'AWS_SECRET_ACCESS_KEY': 'your_secret',
            'AWS_REGION': 'ap-northeast-2',
            'S3_BUCKET_NAME': 'mlops-pipeline-jeb'
        }
    )

    task_temp >> task_pm10 >> task_eda >> task_upload_s3