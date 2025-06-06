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
        ("temperature", temp_dates, "./src/data/processed/temperature", "results/temperature"),
        ("pm10", pm10_dates, "./src/data/processed/pm10", "results/pm10")
    ]

    for category, dates, local_dir, s3_prefix in targets:
        for date_str in dates:
            local_path = os.path.join(local_dir, f"{date_str}.csv")
            if not os.path.exists(local_path):
                print(f"❌ Local file not found: {local_path}")
                continue

            try:
                df_local = pd.read_csv(local_path)
            except Exception as err:
                failed.append((local_path, f"Read error: {err}"))
                continue

            s3_key = f"{s3_prefix}/date={date_str[:7]}/{date_str}.csv"
            need_upload = True

            # S3 파일이 존재하고 내용이 동일하면 skip
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
                df_s3 = pd.read_csv(io.BytesIO(obj['Body'].read()))
                if df_local.equals(df_s3):
                    need_upload = False
                    skipped += 1
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    failed.append((s3_key, str(e)))
                    continue  # 다른 오류는 실패로 간주

            # 내용이 다르거나 처음 업로드라면 put
            if need_upload:
                try:
                    csv_buffer = io.StringIO()
                    df_local.to_csv(csv_buffer, index=False)
                    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=csv_buffer.getvalue())
                    uploaded += 1
                except Exception as err:
                    failed.append((s3_key, f"Upload error: {err}"))

    print("\n==== [S3 Upload Summary] ====")
    print(f"Uploaded: {uploaded}")
    print(f"Skipped : {skipped}")
    print(f"Failed  : {len(failed)}")

    if failed:
        for f, msg in failed:
            print(f"  - {f}: {msg}")

def main(temp_dates, pm10_dates):
    return upload_to_s3(temp_dates, pm10_dates)