import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from datetime import datetime
from scripts import loadTemp, loadAD, eda
import boto3
from dotenv import load_dotenv

load_dotenv()
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'ap-northeast-2')

def upload_to_s3(local_file_path, s3_bucket, s3_prefix):
    s3 = boto3.client('s3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

    today = datetime.today()
    partition = today.strftime('%Y-%m-%d') 
    filename = os.path.basename(local_file_path)
    s3_key = f"{s3_prefix}/date={partition}/{filename}"

    s3.upload_file(local_file_path, s3_bucket, s3_key)
    print(f"Data processing and S3 upload completed! (s3://{s3_bucket}/{s3_key})")


def main():
    print("Loading temperature data...")
    loadTemp.main()

    print("Loading PM10 data...")
    loadAD.main()

    print("Running EDA and saving results...")
    eda.main()

    s3_bucket = 'mlops-pipeline-jeb'
    from config import TA_MODEL_FILE, PM10_MODEL_FILE

    upload_to_s3(TA_MODEL_FILE, s3_bucket, 'results/temperature')
    upload_to_s3(PM10_MODEL_FILE, s3_bucket, 'results/pm10')

    print("Data processing and S3 upload completed successfully!")

if __name__ == "__main__":
    main()