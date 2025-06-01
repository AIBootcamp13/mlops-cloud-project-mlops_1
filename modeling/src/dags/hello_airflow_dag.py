from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ryan',
    'depends_on_past': False,
    'start_date': datetime(2025,5,27),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'hello_airflow_dag',
    default_args = default_args,
    description="our first time practice airflow",
    schedule_interval=timedelta(days=1),
)

sentence = "hello airflow dag. fast campus lecture! we can do it."

def print_word(word):
    print(word)

prev_task = None
for i, word in enumerate(sentence.split()):
    task  = PythonOperator(
        task_id=f'print_world_{i}',
        python_callable=print_word,
        op_kwargs={'word': word},
        dag=dag,
    )

    if prev_task:
        prev_task >> task

    prev_task = task