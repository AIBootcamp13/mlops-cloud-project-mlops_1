import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import io
import os
import sys
import glob

from datetime import datetime
import pytz
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError

def get_outputs():
    outputs_temperature = ["TA_AVG", "TA_MAX", "TA_MIN"]
    outputs_PM = ["PM10_MIN", "PM10_MAX", "PM10_AVG"]
    return outputs_temperature, outputs_PM

def get_scaler(data_path, outputs):
    df = pd.read_csv(data_path)
    features = df[outputs].values
    scaler = MinMaxScaler()
    scaler.fit_transform(features)
    return scaler

def temperature_to_df(results, outputs):
    return pd.DataFrame(
        data=[[results[outputs[0]], results[outputs[1]], results[outputs[2]]]],
        columns=outputs
    )

def PM_to_df(results, outputs):
    return pd.DataFrame(
        data=[[results[outputs[0]], results[outputs[1]], results[outputs[2]]]],
        columns=outputs
    )

def download_pm10_from_s3(data_root_path):
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
    ymd = datetime.now(kst).strftime('%Y%m%d')

    data_files = glob.glob(os.path.join(data_root_path, f'{ymd}_*_PM10_data.csv'))

    if data_files:
        print(f'{ymd}_PM10_data.csv Exists!!')
        return

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
    new_df.to_csv(os.path.join(data_root_path, f'{now}_PM10_data.csv'), index=False)

CFG = {
    'WINDOW_SIZE': 7,
    'EPOCHS': 5,
}