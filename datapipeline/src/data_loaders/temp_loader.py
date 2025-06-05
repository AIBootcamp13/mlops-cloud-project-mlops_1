import os
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from src.config.settings import TA_RAW_FILE, TA_PROCESSED_FILE

def download_ta_data(api_key, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php?tm1=19040401&obs=TA&stn=108&help=0&mode=0&authKey={api_key}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"[DOWNLOAD] Temp Data saved to {save_path}")

def preprocess_ta_data(file_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(file_path, 'r', encoding='cp949') as f:
        data_str = ''.join(line for line in f if not line.startswith('#'))

    df = pd.read_csv(StringIO(data_str), sep=r'\s+', header=None, engine='python')
    df_ta = df.iloc[:, [0, 10, 11, 13]].copy()
    df_ta.columns = ['date', 'TA_AVG', 'TA_MAX', 'TA_MIN']
    df_ta['date'] = pd.to_datetime(df_ta['date'].astype(str), errors='coerce')
    df_ta = df_ta.astype({'TA_AVG': 'float32', 'TA_MAX': 'float32', 'TA_MIN': 'float32'})
    df_ta.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[SAVE] Processed Temp data saved to {output_path}")
    return df_ta

def main():
    load_dotenv()
    api_key = os.getenv('TEMP_API_KEY')
    download_ta_data(api_key, TA_RAW_FILE)
    preprocess_ta_data(TA_RAW_FILE, TA_PROCESSED_FILE)