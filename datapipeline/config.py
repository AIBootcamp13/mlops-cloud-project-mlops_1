import os

# config.py가 있는 datapipeline 디렉토리 기준
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # datapipeline/

# 데이터 디렉토리
DATA_DIR = os.path.join(BASE_DIR, 'data')             # datapipeline/data
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# 파일 경로
TA_RAW_FILE = os.path.join(RAW_DIR, 'data1.csv')
TA_PROCESSED_FILE = os.path.join(PROCESSED_DIR, 'TA_data.csv')

PM10_RAW_FILE = os.path.join(RAW_DIR, 'data2.csv')
PM10_PROCESSED_FILE = os.path.join(PROCESSED_DIR, 'PM10_data.csv')

TA_MODEL_FILE = os.path.join(PROCESSED_DIR, 'modeling_TA_data.csv')
PM10_MODEL_FILE = os.path.join(PROCESSED_DIR, 'modeling_PM10_data.csv')