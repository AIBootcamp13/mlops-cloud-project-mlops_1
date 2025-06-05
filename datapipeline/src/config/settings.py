import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

TA_RAW_FILE = os.path.join(RAW_DIR, 'data1.csv')
TA_PROCESSED_FILE = os.path.join(PROCESSED_DIR, 'TA_data.csv')
TA_MODEL_FILE = os.path.join(PROCESSED_DIR, 'modeling_TA_data.csv')

PM10_RAW_FILE = os.path.join(RAW_DIR, 'data2.csv')
PM10_PROCESSED_FILE = os.path.join(PROCESSED_DIR, 'PM10_data.csv')
PM10_MODEL_FILE = os.path.join(PROCESSED_DIR, 'modeling_PM10_data.csv')