# 맨 처음 s3에 데이터 올려져있지 않을 떄 초기 시행(처음 한번만)
import sys
import os
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# src 경로를 Python path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 이제 import 가능
from src.data_processors.eda_processor import run_full_eda
from src.cloud.s3_uploader import upload_to_s3

# 실행 함수 정의
def run_and_upload():
    temp_dates, pm10_dates = run_full_eda()
    upload_to_s3(temp_dates, pm10_dates)

# DAG 설정
default_args = {
    'owner': 'eunbyul',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

dag = DAG(
    dag_id='init_full_eda_dag',
    default_args=default_args,
    schedule_interval=None,  # 수동 실행 전용
    catchup=False,
    tags=['mlops', 'init']
)

# Task 정의
task_full_eda_upload = PythonOperator(
    task_id='run_full_eda_and_upload',
    python_callable=run_and_upload,
    dag=dag
)