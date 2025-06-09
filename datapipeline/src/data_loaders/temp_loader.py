import os
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from datetime import date

# 환경 변수 로드
load_dotenv()

def download_ta_data(api_key: str) -> bytes:
    url = (
        "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"
        "?tm1=19040401&obs=TA&stn=108&help=0&mode=0"
        f"&authKey={api_key}"
    )
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        raise RuntimeError(f"기온 데이터 요청 실패: {e}\n요청 URL: {url}")

def preprocess_ta_data(binary_data: bytes) -> pd.DataFrame:
    data_str = binary_data.decode('cp949')
    data_str = '\n'.join(line for line in data_str.splitlines() if not line.startswith('#'))

    df = pd.read_csv(StringIO(data_str), sep=r'\s+', header=None, engine='python')
    
    # 0: 날짜, 10: 평균기온, 11: 최고기온, 13: 최저기온
    df_ta = df.iloc[:, [0, 10, 11, 13]].copy()
    df_ta.columns = ['date', 'TA_AVG', 'TA_MAX', 'TA_MIN']
    
    df_ta['date'] = pd.to_datetime(df_ta['date'].astype(str), errors='coerce')
    df_ta = df_ta.astype({'TA_AVG': 'float32', 'TA_MAX': 'float32', 'TA_MIN': 'float32'})

    today = pd.to_datetime(date.today())
    df_ta = df_ta[df_ta['date'] < today]

    if df_ta.empty:
        raise ValueError("기온 데이터가 비어 있습니다. API 응답 또는 날짜 필터링 확인 필요.")

    return df_ta

def run_temp_preprocessing() -> pd.DataFrame:
    api_key = os.getenv('TEMP_API_KEY')
    if not api_key:
        raise ValueError("TEMP_API_KEY가 .env에 설정되지 않았습니다.")
    
    raw_data = download_ta_data(api_key)
    return preprocess_ta_data(raw_data)