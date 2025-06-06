import json
import requests
from airflow.models import Variable

def notify_slack(context):
    webhook_url = Variable.get("SLACK_WEBHOOK_URL")

    dag_id = context.get("dag").dag_id
    task_instance = context.get("task_instance")
    task_id = task_instance.task_id
    execution_date = context.get("execution_date")
    status = task_instance.state
    log_url = task_instance.log_url

    if status == 'failed':
        message = {
            "text": f":x: *Task Failed!*\n"
                    f"*DAG*: `{dag_id}`\n"
                    f"*Task*: `{task_id}`\n"
                    f"*Execution Time*: `{execution_date}`\n"
                    f"*Log URL*: <{log_url}|View Logs>"
        }

    elif status == 'success':
        if task_id == 'load_temperature_data':
            text = ":thermometer: 기온 데이터 수집 완료!"
        elif task_id == 'load_pm10_data':
            text = ":cloud: 미세먼지 데이터 수집 완료!"
        elif task_id == 'run_eda':
            text = ":bar_chart: EDA 완료!"
        elif task_id == 'upload_to_s3_docker':
            text = ":package: S3 업로드 완료!"
        else:
            text = f":white_check_mark: `{dag_id}` - `{task_id}` 성공!"

        message = {"text": text}

    else:
        message = {
            "text": f":grey_question: `{dag_id}` - `{task_id}` Task 상태: `{status}`"
        }

    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(webhook_url, data=json.dumps(message), headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Slack notification failed: {e}")
