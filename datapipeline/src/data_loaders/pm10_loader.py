import os
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from datetime import date

# .env 로드
load_dotenv()

STATION_ID = 108
START_TIME = "200804280000"

def download_pm10_data(api_key: str) -> bytes:
    url = (
        f"https://apihub.kma.go.kr/api/typ01/url/kma_pm10.php"
        f"?tm1={START_TIME}&stn={STATION_ID}&authKey={api_key}"
    )
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise RuntimeError(f"PM10 데이터 요청 실패: {e}\n요청 URL: {url}")

def preprocess_pm10_data(binary_data: bytes) -> pd.DataFrame:
    data_str = binary_data.decode('cp949')
    data_str = '\n'.join(line for line in data_str.splitlines() if not line.startswith('#') and line.strip())

    df = pd.read_csv(StringIO(data_str), sep=',', header=None, skipinitialspace=True, on_bad_lines='skip', engine='python')
    
    # 0: timestamp (YYYYMMDDHHMM), 2: PM10 수치
    df_ad = df.iloc[:, [0, 2]].copy()
    df_ad.columns = ['timestamp', 'PM10']
    
    df_ad['timestamp'] = df_ad['timestamp'].astype(str).str.extract(r'(\d{12})')[0]
    df_ad['date'] = pd.to_datetime(df_ad['timestamp'], format='%Y%m%d%H%M', errors='coerce').dt.date
    df_ad['PM10'] = pd.to_numeric(df_ad['PM10'], errors='coerce', downcast='float')

    # 일별 평균/최소/최대
    df_pm10 = df_ad.groupby('date', as_index=False)['PM10'].agg(PM10_MIN='min', PM10_MAX='max', PM10_AVG='mean')
    df_pm10 = df_pm10.round(1)
    df_pm10['date'] = pd.to_datetime(df_pm10['date'].astype(str))

    # 오늘 날짜 제거
    today = pd.to_datetime(date.today())
    df_pm10 = df_pm10[df_pm10['date'] < today]

    if df_pm10.empty:
        raise ValueError("PM10 데이터가 비어 있습니다. API 응답 또는 날짜 필터링 확인 필요.")

    return df_pm10

def run_pm10_preprocessing() -> pd.DataFrame:
    api_key = os.getenv('AD_API_KEY')
    if not api_key:
        raise ValueError("AD_API_KEY가 .env에 설정되지 않았습니다.")
    
    raw_data = download_pm10_data(api_key)
    return preprocess_pm10_data(raw_data)