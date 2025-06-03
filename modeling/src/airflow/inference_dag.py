import os
import shutil
import sys

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(project_path)
sys.path.append(project_path)

from datetime import timedelta
from textwrap import dedent
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from modeling.src.airflow.tasks.inference import inference
from modeling.src.airflow.tasks.is_model_drift import is_model_drift

with DAG(
    "anomaly_model_inference",
    default_args = {
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description="anomaly model inference",
    schedule=timedelta(days=7),
    start_date=pendulum.now("UTC").subtract(days=1),
    catchup=False,
) as dag:
    inference_task = PythonOperator(
        task_id='inference_task',
        python_callable=inference,
        op_kwargs={'project_path': project_path},
    )

    is_model_drift_task = ShortCircuitOperator(
        task_id="is_model_drift_task",
        python_callback=is_model_drift,
        op_kwargs={'project_path': project_path},
    )

    train_trigger_task = TriggerDagRunOperator(
        task_id="train_trigger_task",
        trigger_dag_id="anomaly_model_train",
    )

    inference_task >> is_model_drift_task >> train_trigger_task

if __name__ == "__main__":
    if is_model_drift(project_path) == True:
        inference(project_path)