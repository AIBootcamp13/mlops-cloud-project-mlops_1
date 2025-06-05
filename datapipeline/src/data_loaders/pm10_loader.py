import os
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from src.config.settings import PM10_RAW_FILE, PM10_PROCESSED_FILE

STATION_ID = 108
START_TIME = "200804280000"

def download_pm10_data(api_key, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_pm10.php?tm1={START_TIME}&stn={STATION_ID}&authKey={api_key}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"[DOWNLOAD] PM10 Data saved to {save_path}")

def preprocess_pm10_data(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, 'r', encoding='cp949') as f:
        data_str = ''.join(line for line in f if not line.startswith('#') and line.strip())

    df = pd.read_csv(StringIO(data_str), sep=',', header=None, skipinitialspace=True, on_bad_lines='skip', engine='python')
    df_ad = df.iloc[:, [0, 2]].copy()
    df_ad.columns = ['timestamp', 'PM10']
    df_ad['timestamp'] = df_ad['timestamp'].astype(str).str.extract(r'(\d{12})')[0]
    df_ad['date'] = pd.to_datetime(df_ad['timestamp'], format='%Y%m%d%H%M', errors='coerce').dt.date
    df_ad['PM10'] = pd.to_numeric(df_ad['PM10'], errors='coerce', downcast='float')

    df_pm10 = df_ad.groupby('date', as_index=False)['PM10'].agg(['min', 'max', 'mean'])
    df_pm10.columns = ['date', 'PM10_MIN', 'PM10_MAX', 'PM10_AVG']
    df_pm10 = df_pm10.round(1)
    df_pm10['date'] = pd.to_datetime(df_pm10['date'].astype(str))
    df_pm10.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[SAVE] Processed PM10 data saved to {output_path}")
    return df_pm10

def main():
    load_dotenv()
    api_key = os.getenv('AD_API_KEY')
    download_pm10_data(api_key, PM10_RAW_FILE)
    preprocess_pm10_data(PM10_RAW_FILE, PM10_PROCESSED_FILE)