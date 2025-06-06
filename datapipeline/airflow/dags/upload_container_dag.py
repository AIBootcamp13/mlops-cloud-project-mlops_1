from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from datapipeline.airflow.utils.slack_notifier import notify_slack

default_args = {
    'owner': 'eunbyul',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': notify_slack,
    'on_success_callback': notify_slack,
}

with DAG(
    dag_id='upload_s3_docker_dag',
    default_args=default_args,
    description='Run S3 upload script via Docker',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'docker'],
) as dag:

    run_upload_container = DockerOperator(
        task_id='run_upload_container',
        image='294063201644.dkr.ecr.ap-northeast-2.amazonaws.com/upload-script:latest',
        api_version='auto',
        auto_remove=True,
        command='sh -c "export PYTHONPATH=/app && python src/cloud/upload_script.py --temp_dates=2025-06-01,2025-06-02 --pm10_dates=2025-06-01,2025-06-02"',
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        environment={
            'AWS_ACCESS_KEY_ID': '{{ var.value.AWS_ACCESS_KEY_ID }}',
            'AWS_SECRET_ACCESS_KEY': '{{ var.value.AWS_SECRET_ACCESS_KEY }}',
            'AWS_REGION': '{{ var.value.AWS_REGION }}',
            'S3_BUCKET_NAME': '{{ var.value.S3_BUCKET_NAME }}',
        },
        on_failure_callback=notify_slack,
        on_success_callback=notify_slack,
    )