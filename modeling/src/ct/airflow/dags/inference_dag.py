from datetime import timedelta
from textwrap import dedent
import pendulum

from airflow import DAG
from airflow.providers.standard.operators.python import PythonVirtualenvOperator
from tasks.inference import inference

with DAG(
    "inference",
    default_args={
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description="inference",
    schedule=timedelta(days=7),
    start_date=pendulum.now("UTC").subtract(days=1),
    catchup=False,
) as dag:
    
    inference_task = PythonVirtualenvOperator(
        task_id="inference_task",
        requirements=[
            'numpy==1.24.4',
            'torch==2.4.1',
            'pandas==2.0.3',
            'matplotlib==3.7.5',
            'scikit-learn==1.3.2'
        ],
        python_callable=inference,
        python_version="3.10"
    )

    inference_task

if __name__ == "__main__":
    inference()