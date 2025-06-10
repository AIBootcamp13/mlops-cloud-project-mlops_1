import os
import shutil
import sys

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(project_path)
sys.path.append(project_path)

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

from modeling.src.train.train import run_temperature_train

default_args = {
    'owner': 'lmw',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=15),
}

dag = DAG(
    'tutorial',
    default_args=default_args,
    description='a simple dag for lstm model training and selection',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

def preprocessing_data(**kwargs):
    data = pd.read_csv(os.path.join(project_path, 'data/TA_data.csv'))
    
    ti = kwargs['ti']
    ti.xcom_push(key='data', value=data.to_json())

def train_model(model_name, **kwargs):
    ti = kwargs['ti']
    data = pd.read_json(ti.xcom_pull(key='data', task_ids='preprocessing_data'))

    model_root_path = os.path.join(project_path, 'models/tmp')
    data_root_path = os.path.join(project_path, 'data')

    _, val_loss = run_temperature_train(data_root_path, model_root_path, 64, model_name)

    ti.xcom_push(key=f'val_loss_{model_name}', value=val_loss)

def select_best_model(**kwargs):
    ti = kwargs['ti']
    val_loss_multi_output_lstm = ti.xcom_pull(key='val_loss_multi_output_lstm', task_ids='train_multi_output_lstm')
    val_loss_multi_output_stacked_lstm = ti.xcom_pull(key='val_loss_multi_output_stacked_lstm', task_ids='train_multi_output_stacked_lstm')

    if val_loss_multi_output_lstm > val_loss_multi_output_stacked_lstm:
        best_model = 'multi_output_stacked_lstm'
    else:
        best_model = 'multi_output_lstm'

    shutil.copy2(os.path.join(project_path, 'models/tmp', f"{best_model}.pth"), os.path.join(project_path, 'models', f"{best_model}.pth"))
    shutil.rmtree(os.path.join(project_path, 'models/tmp'))
    
    print(best_model)
    return best_model

with dag:
    t1 = PythonOperator(
        task_id='preprocessing_data',
        python_callable=preprocessing_data,
    )

    t2 = PythonOperator(
        task_id='train_multi_output_lstm',
        python_callable=train_model,
        op_kwargs={'model_name': 'multi_output_lstm'},
    )

    # t3 = PythonOperator(
    #     task_id='train_multi_output_stacked_lstm',
    #     python_callable=train_model,
    #     op_kwargs={'model_name': 'multi_output_stacked_lstm'},
    # )

    # t4 = PythonOperator(
    #     task_id='select_best_model',
    #     python_callable=select_best_model,
    # )

    # t1 >> [t2, t3] >> t4
    t1 >> t2