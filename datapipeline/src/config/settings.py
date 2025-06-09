import os
from dotenv import load_dotenv

load_dotenv()

# S3 환경 변수
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

# S3 업로드 경로 (폴더 형식)
S3_TEMP_FOLDER = "results/temperature/"
S3_PM10_FOLDER = "results/pm10/"

# API Key
TEMP_API_KEY = os.getenv("TEMP_API_KEY")
AD_API_KEY = os.getenv("AD_API_KEY")

# Slack
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")