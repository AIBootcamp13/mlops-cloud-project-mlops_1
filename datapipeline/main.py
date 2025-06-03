import sys
import os
import glob
import boto3
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from scripts import loadTemp, loadAD, eda

load_dotenv()
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'ap-northeast-2')

# S3 클라이언트는 한 번만 생성 (전체 코드에서 재사용)
s3 = boto3.client('s3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def get_existing_s3_keys(s3, bucket, prefix):
    keys = set()
    continuation_token = None

    # S3에 있는 전체 파일 키 목록을 한 번에 가져오기 (기존 head_object 반복 호출 대신)
    while True:
        if continuation_token:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=continuation_token)
        else:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        contents = response.get('Contents', [])
        for obj in contents:
            keys.add(obj['Key'])  # S3 키만 저장

        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break

    return keys

def upload_to_s3(s3, local_file_path, s3_bucket, s3_key):
    try:
        s3.upload_file(local_file_path, s3_bucket, s3_key)
        print(f"[UPLOAD] S3 upload completed: s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        print(f"[ERROR] Failed to upload {local_file_path}: {e}")

def main():
    print("Loading temperature data...")
    loadTemp.main()

    print("Loading PM10 data...")
    loadAD.main()

    print("Running EDA and saving results...")
    eda.main()

    s3_bucket = 'mlops-pipeline-jeb'

    print("Fetching existing S3 keys...")
    temp_s3_keys = get_existing_s3_keys(s3, s3_bucket, 'results/temperature')
    pm10_s3_keys = get_existing_s3_keys(s3, s3_bucket, 'results/pm10')

    temp_files = glob.glob('./data/processed/temperature/*.csv')
    for file in temp_files:
        filename = os.path.basename(file)
        date_month = os.path.splitext(filename)[0][:7]
        s3_key = f"results/temperature/date={date_month}/{filename}"

        if s3_key not in temp_s3_keys:
            upload_to_s3(s3, file, s3_bucket, s3_key)
        else:
            print(f"[SKIP] Already uploaded: s3://{s3_bucket}/{s3_key}")

    pm10_files = glob.glob('./data/processed/pm10/*.csv')
    for file in pm10_files:
        filename = os.path.basename(file)
        date_month = filename.replace('.csv', '')[:7]
        s3_key = f"results/pm10/date={date_month}/{filename}"

        if s3_key not in pm10_s3_keys:
            upload_to_s3(s3, file, s3_bucket, s3_key)
        else:
            print(f"[SKIP] Already uploaded: s3://{s3_bucket}/{s3_key}")

    print("Data processing and S3 upload completed successfully!")

if __name__ == "__main__":
    main()