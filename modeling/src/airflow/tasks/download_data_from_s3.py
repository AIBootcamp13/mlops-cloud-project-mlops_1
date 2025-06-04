import io
import os
import sys

from datetime import datetime
import pytz
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError

def download_pm10_from_s3(data_root_path):
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst).strftime('%Y%m%d_%H%M%S')

    bucket_name = 'mlops-pipeline-jeb'

    prefix = 'results/pm10/'

    data_download_path = os.path.join(data_root_path, f"s3data/{now}", bucket_name)
    os.makedirs(data_root_path, exist_ok=True)
    os.makedirs(data_download_path, exist_ok=True)

    s3 = boto3.client('s3')
   
    merged_df_list = []
    continuation_token = None

    while True:
        if continuation_token:
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                ContinuationToken=continuation_token
            )
        else:
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )

        if 'Contents' not in response:
            break

        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.csv'):
                local_path = os.path.join(data_download_path, key)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                try:
                    s3.download_file(bucket_name, key, local_path)
                    print(f"Downloading: s3://{bucket_name}/{key} -> {local_path}")
                except NoCredentialsError:
                    print("AWS credentials not found. Check your S3 Access Key")
                    return

                try:
                    df = pd.read_csv(local_path)
                    df['source_file'] = key
                    merged_df_list.append(df)
                except Exception as e:
                    print(f"Error reading {key}: {e}")
        
        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break

    if merged_df_list:
        merged_df = pd.concat(merged_df_list, ignore_index=True)
        merged_df.to_csv(os.path.join(data_root_path, f'{now}_PM10_data.csv'))
    else:
        print("No CSVs were successfully read.")