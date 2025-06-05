import os
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import io
import pandas as pd

def upload_to_s3(temp_dates, pm10_dates):
    load_dotenv()

    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'ap-northeast-2')
    S3_BUCKET = os.getenv('S3_BUCKET_NAME', 'mlops-pipeline-jeb')

    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

    uploaded, skipped, failed = 0, 0, []
    targets = [
        ("temperature", temp_dates, "results/temperature"),
        ("pm10", pm10_dates, "results/pm10")
    ]

    for category, dates, prefix in targets:
        for date_str in dates:
            s3_key_local = f"results/{category}/date={date_str[:7]}/{date_str}.csv"

            # S3에서 원본 읽기
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key_local)
                df = pd.read_csv(io.BytesIO(obj['Body'].read()))
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    print(f"File does not exist in S3: {s3_key_local}")
                    continue
                else:
                    raise

            # 업로드 대상 경로 만들기
            month_partition = date_str[:7]
            s3_key = f"{prefix}/date={month_partition}/{date_str}.csv"

            # 이미 존재하면 skip
            try:
                s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
                skipped += 1
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    try:
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=csv_buffer.getvalue())
                        uploaded += 1
                    except Exception as err:
                        failed.append((s3_key, str(err)))
                else:
                    failed.append((s3_key, str(e)))

    print("\n==== [S3 Upload Summary] ====")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {len(failed)}")
    
    if failed:
        for f, msg in failed:
            print(f"  - {f}: {msg}")

"""
# 전체 날짜 업로드 버전
import glob

def upload_all_files():
    categories = ['temperature', 'pm10']
    total_uploaded = 0
    total_skipped = 0
    total_failed = 0

    for category in categories:
        file_list = glob.glob(f"./src/data/processed/{category}/*.csv")
        for file_path in file_list:
            file_name = os.path.basename(file_path)
            s3_key_local = f"results/{category}/date={date_str[:7]}/{date_str}.csv"

            if os.path.exists(file_path):
                success = upload_file(file_path, s3_key)
                if success:
                    total_uploaded += 1
                else:
                    total_failed += 1
            else:
                total_skipped += 1

    print("\n==== [S3 Upload Summary] ====")
    print(f"Uploaded: {total_uploaded}")
    print(f"Skipped : {total_skipped}")
    print(f"Failed  : {total_failed}")
"""

def main(temp_dates, pm10_dates):
    return upload_to_s3(temp_dates, pm10_dates)