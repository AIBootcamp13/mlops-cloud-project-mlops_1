import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from config import TA_RAW_FILE, TA_PROCESSED_FILE

# 기상청 기상 데이터 다운
def download_ta_data(api_key, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php?tm1=19040401&obs=TA&stn=108&help=0&mode=0&authKey={api_key}"
    response = requests.get(url)

    with open(save_path, 'wb') as f:
        f.write(response.content)

# 기상 데이터 로드 및 전처리
def preprocess_ta_data(file_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(file_path, 'r', encoding='cp949') as f:
        lines = f.readlines()
    data_lines = [line for line in lines if not line.startswith('#')]
    data_str = ''.join(data_lines)

    df = pd.read_csv(StringIO(data_str), sep=r'\s+', header=None)
    
    df_ta = df.iloc[:, [0, 10, 11, 13]].copy()
    df_ta.columns = ['date', 'TA_AVG', 'TA_MAX', 'TA_MIN']
    df_ta['date'] = pd.to_datetime(df_ta['date'].astype(str), errors='coerce')

    df_ta.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Save completed: {output_path}")
    
    return df_ta

def main():
    load_dotenv()
    api_key = os.getenv('TEMP_API_KEY')

    download_ta_data(api_key, TA_RAW_FILE)
    preprocess_ta_data(TA_RAW_FILE, TA_PROCESSED_FILE)