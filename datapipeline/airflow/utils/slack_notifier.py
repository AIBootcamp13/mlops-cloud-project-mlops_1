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

    # 실패 시 상세 메시지
    if status == 'failed':
        message = {
            "text": f":x: *Task Failed!*\n"
                    f"*DAG*: `{dag_id}`\n"
                    f"*Task*: `{task_id}`\n"
                    f"*Execution Time*: `{execution_date}`\n"
                    f"*Log URL*: <{log_url}|View Logs>"
        }

    # 성공 시 간단 메시지
    elif status == 'success':
        message = {
            "text": f":white_check_mark: `{dag_id}` DAG 성공 완료!"
        }

    # 기타 상태 (optional)
    else:
        message = {
            "text": f":grey_question: `{dag_id}` - `{task_id}` Task 상태: `{status}`"
        }

    headers = {"Content-Type": "application/json"}
    requests.post(webhook_url, data=json.dumps(message), headers=headers)