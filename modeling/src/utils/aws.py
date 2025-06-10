import os
import glob

from datetime import datetime
import pytz
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError

import os
import glob

from datetime import datetime
import pytz
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError

def fast_download_data_from_s3(data_root_path, data_name):
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
    ymd = datetime.now(kst).strftime('%Y%m%d')

    data_files = glob.glob(os.path.join(data_root_path, f'{ymd}_*_{data_name}_data.csv'))

    if data_files:
        print(f'{ymd}_{data_name}_data.csv Exists!!')
        return

    bucket_name = 'mlops-pipeline-jeb'

    prefix = f'results/{data_name}/'

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
        count = 0
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.csv'):
                local_path = os.path.join(data_download_path, key)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                try:
                    count += 1
                    s3.download_file(bucket_name, key, local_path)
                    print(f"Downloading: s3://{bucket_name}/{key} -> {local_path}")
                except NoCredentialsError:
                    print("AWS credentials not found. Check your S3 Access Key")
                    return

                try:
                    df = pd.read_csv(local_path)
                    # df['source_file'] = key
                    merged_df_list.append(df)
                except Exception as e:
                    print(f"Error reading {key}: {e}")
            if count > 250:
                break
            
        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break

    if not merged_df_list:
        print("no csv were successfully read.")
        return

    new_df = pd.concat(merged_df_list, ignore_index=True)
    new_df.to_csv(os.path.join(data_root_path, f'{now}_{data_name}_data.csv'), index=False)


def download_data_from_s3(data_root_path, data_name):
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
    ymd = datetime.now(kst).strftime('%Y%m%d')

    data_files = glob.glob(os.path.join(data_root_path, f'*_{data_name}_data.csv'))

    if data_files:
        print(f'{data_name}_data.csv Exists!!')
        return

    bucket_name = 'mlops-pipeline-jeb'

    prefix = f'results/{data_name}/'

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
                    # df['source_file'] = key
                    merged_df_list.append(df)
                except Exception as e:
                    print(f"Error reading {key}: {e}")
        
        if response.get('IsTruncated'):
            continuation_token = response['NextContinuationToken']
        else:
            break

    if not merged_df_list:
        print("no csv were successfully read.")
        return

    new_df = pd.concat(merged_df_list, ignore_index=True)
    new_df.to_csv(os.path.join(data_root_path, f'{now}_{data_name}_data.csv'), index=False)
