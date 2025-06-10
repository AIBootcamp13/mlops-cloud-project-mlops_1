import io
import os
import glob
import sys

from dotenv import load_dotenv
load_dotenv()

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(project_path)
sys.path.append(project_path)

from datetime import timedelta
from textwrap import dedent
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from modeling.src.inference.anomaly import inference
from modeling.src.inference.anomaly import is_model_drift

with DAG(
    "inference_anomaly_temperature_on_airflow",
    default_args = {
        'owner': 'lmw',
        'depends_on_past': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description="inference anomaly temperature on airflow",
    schedule=timedelta(days=7),
    start_date=pendulum.now("UTC").subtract(days=1),
    catchup=False,
) as dag:
    inference_task = PythonOperator(
        task_id='inference_task',
        python_callable=inference,
        op_kwargs={
            'project_path': project_path,
            'data_name': 'temperature', 
            'model_name': 'temperature'
        },
    )

    is_model_drift_task = ShortCircuitOperator(
        task_id="is_model_drift_task",
        python_callable=is_model_drift,
        op_kwargs={
            'project_path': project_path,
            'model_name': 'temperature'
        },
    )

    train_anomaly_temperature_trigger_task = TriggerDagRunOperator(
        task_id="train_anomaly_temperature_trigger_task",
        trigger_dag_id="train_anomaly_temperature_on_airflow",
    )

    train_temperature_trigger_task = TriggerDagRunOperator(
        task_id="train_temperature_trigger_task",
        trigger_dag_id="train_temperature_on_airflow",
    )

    inference_task >> is_model_drift_task >> train_anomaly_temperature_trigger_task >> train_temperature_trigger_task