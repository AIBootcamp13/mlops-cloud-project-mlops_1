import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from config import PM10_RAW_FILE, PM10_PROCESSED_FILE

# 기상청 황사 데이터 다운
def download_pm10_data(api_key, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_pm10.php?tm1=200804280000&stn=108&authKey={api_key}"
    response = requests.get(url)
    
    with open(save_path, 'wb') as f:
        f.write(response.content)

# 황사 데이터 로드 및 전처리
def preprocess_pm10_data(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r', encoding='cp949') as f:
        lines = f.readlines()
    data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
    data_str = '\n'.join(data_lines)
    
    df = pd.read_csv(StringIO(data_str), sep=',', header=None, skipinitialspace=True, on_bad_lines='skip', engine='python')

    if df.shape[1] >= 3:
        df_ad = df.iloc[:, [0, 2]].copy()
        df_ad.columns = ['timestamp', 'PM10']
    else:
        raise ValueError("데이터 컬럼이 부족합니다.")

    df_ad['timestamp'] = df_ad['timestamp'].astype(str).str.extract(r'(\d{12})')[0]
    df_ad['date'] = pd.to_datetime(df_ad['timestamp'], format='%Y%m%d%H%M', errors='coerce').dt.date
    df_ad['PM10'] = pd.to_numeric(df_ad['PM10'], errors='coerce')

    # 일별 집계
    df_pm10 = df_ad.groupby('date')['PM10'].agg(['min', 'max', 'mean']).reset_index()
    df_pm10.columns = ['date', 'PM10_MIN', 'PM10_MAX', 'PM10_AVG']
    df_pm10 = df_pm10.round(1)
    df_pm10['date'] = pd.to_datetime(df_pm10['date'].astype(str))

    df_pm10.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Save completed: {output_path}")
    
    return df_pm10

def main():
    load_dotenv()
    api_key = os.getenv('AD_API_KEY')

    download_pm10_data(api_key, PM10_RAW_FILE)
    preprocess_pm10_data(PM10_RAW_FILE, PM10_PROCESSED_FILE)