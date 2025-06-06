from datetime import timedelta
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
    description='Upload processed weather data to S3 via Docker container',
    schedule_interval=None,  # 필요 시 '@daily' 등으로 변경
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'docker'],
) as dag:

    run_upload_container = DockerOperator(
        task_id='run_upload_container',
        image='294063201644.dkr.ecr.ap-northeast-2.amazonaws.com/upload-script:latest',
        api_version='auto',
        auto_remove=True,
        command=(
            'sh -c "export PYTHONPATH=/opt/airflow && '
            'python src/cloud/upload_script.py '
            '--temp_dates={{ macros.ds_add(ds, -1) }} '
            '--pm10_dates={{ macros.ds_add(ds, -1) }}"'
        ),
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        environment={
            'AWS_ACCESS_KEY_ID': '{{ var.value.AWS_ACCESS_KEY_ID }}',
            'AWS_SECRET_ACCESS_KEY': '{{ var.value.AWS_SECRET_ACCESS_KEY }}',
            'AWS_REGION': '{{ var.value.AWS_REGION }}',
            'S3_BUCKET_NAME': '{{ var.value.S3_BUCKET_NAME }}',
        },
    )