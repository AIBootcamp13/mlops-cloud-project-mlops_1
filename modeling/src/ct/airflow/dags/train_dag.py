from datetime import timedelta
from textwrap import dedent
import pendulum

from airflow import DAG
from airflow.providers.standard.operators.python import PythonVirtualenvOperator
from tasks.train import train

with DAG(
    "train",
    default_args={
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description="train",
    schedule=timedelta(days=30),
    start_date=pendulum.now("UTC").subtract(days=1),
    catchup=False,
) as dag:
    
    train_task = PythonVirtualenvOperator(
        task_id="train_task",
        requirements=[
            'numpy==1.24.4',
            'torch==2.4.1',
            'pandas==2.0.3',
            'matplotlib==3.7.5',
            'scikit-learn==1.3.2'
        ],
        python_callable=train,
        python_version="3.10"
    )

    train_task

if __name__ == "__main__":
    train()