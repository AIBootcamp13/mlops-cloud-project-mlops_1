import os
import io
import boto3
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from src.data_processors.eda_processor import run_eda_for_recent_days

# 환경 변수 로드
load_dotenv()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'ap-northeast-2')
S3_BUCKET = os.getenv('S3_BUCKET_NAME', 'mlops-pipeline-jeb')

# boto3 클라이언트 설정
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# 개별 행 단위로 S3에 업로드
def upload_df_to_s3(df: pd.DataFrame, s3_prefix: str, date_col: str = 'date'):
    uploaded, skipped, failed = 0, 0, []

    for _, row in df.iterrows():
        date = row[date_col]
        if pd.isna(date):
            continue

        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        year_month = pd.to_datetime(date).strftime('%Y-%m')

        single_df = pd.DataFrame([row])

        # 소수점 한 자리로 반올림
        numeric_cols = single_df.select_dtypes(include=[np.number]).columns
        single_df[numeric_cols] = single_df[numeric_cols].round(1)

        # S3 저장 키 설정
        s3_key = f"{s3_prefix}/date={year_month}/{date_str}.csv"

        # 중복 확인 후 업로드
        try:
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
                df_existing = pd.read_csv(io.BytesIO(obj['Body'].read()))

                # 컬럼 순서 정렬 및 인덱스 제거 후 비교
                df_existing = df_existing.sort_index(axis=1).reset_index(drop=True)
                single_df = single_df.sort_index(axis=1).reset_index(drop=True)

                if df_existing.equals(single_df):
                    skipped += 1
                    continue

            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    raise

            # 업로드 실행
            csv_buffer = io.StringIO()
            single_df.to_csv(csv_buffer, index=False, float_format='%.1f', columns=single_df.columns)

            s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=csv_buffer.getvalue())
            uploaded += 1

        except Exception as err:
            failed.append((s3_key, str(err)))

    # 업로드 요약 출력
    print("\n==== [S3 Upload Summary] ====")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {len(failed)}")
    if failed:
        for f, msg in failed:
            print(f"  - {f}: {msg}")

# 실제 업로드 실행
def upload_processed_data_to_s3(df_temp: pd.DataFrame, df_pm10: pd.DataFrame):
    upload_df_to_s3(df_temp, s3_prefix="results/temperature", date_col='date')
    upload_df_to_s3(df_pm10, s3_prefix="results/pm10", date_col='date')

# 전체 EDA + 업로드 실행
def upload_to_s3(df_ta: pd.DataFrame, df_pm10: pd.DataFrame):
    df_temp, df_pm10 = run_eda_for_recent_days(df_ta, df_pm10)
    upload_processed_data_to_s3(df_temp, df_pm10)