import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.config.settings import TA_PROCESSED_FILE, TA_MODEL_FILE, PM10_PROCESSED_FILE, PM10_MODEL_FILE

PM10_AVG_THRESHOLD = 90.8
PM10_MAX_THRESHOLD = 160.5

# 프로젝트 기준 base dir 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'temperature')
PM10_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'pm10')
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PM10_DIR, exist_ok=True)

def load_data():
    df_ta = pd.read_csv(TA_PROCESSED_FILE, parse_dates=['date'])
    df_pm10 = pd.read_csv(PM10_PROCESSED_FILE, parse_dates=['date'])
    return df_ta, df_pm10

# 전체 날짜 eda 진행
"""
def preprocess_temperature(df_ta):
    today = pd.to_datetime(date.today())
    df_ta = df_ta[df_ta['date'] < today].replace(-99.0, np.nan)
    df_ta[['TA_AVG', 'TA_MAX', 'TA_MIN']] = df_ta[['TA_AVG', 'TA_MAX', 'TA_MIN']].astype('float32')
    return df_ta

def preprocess_pm10(df_pm10):
    today = pd.to_datetime(date.today())
    return df_pm10[df_pm10['date'] < today]
"""
# 어제 날짜만 eda 진행
def preprocess_temperature(df_ta):
    today = pd.to_datetime(date.today())
    df_ta = df_ta[df_ta['date'] < today].replace(-99.0, np.nan)

    yesterday = pd.to_datetime(date.today() - timedelta(days=1))
    df_ta = df_ta[df_ta['date'] == yesterday]

    df_ta[['TA_AVG', 'TA_MAX', 'TA_MIN']] = df_ta[['TA_AVG', 'TA_MAX', 'TA_MIN']].astype('float32')
    return df_ta

def interpolate_temperature(df_ta):
    return df_ta.set_index('date').interpolate(method='time', limit_direction='both').dropna().reset_index()

def preprocess_pm10(df_pm10):
    today = pd.to_datetime(date.today())
    yesterday = pd.to_datetime(date.today() - timedelta(days=1))

    return df_pm10[(df_pm10['date'] == yesterday) & (df_pm10['date'] < today)]

def process_pm10_outliers(df_pm10):
    df_pm10['PM10_AVG_filtered'] = df_pm10['PM10_AVG'].where(df_pm10['PM10_AVG'] <= PM10_AVG_THRESHOLD)
    df_pm10['PM10_MAX_capped'] = df_pm10['PM10_MAX'].clip(upper=PM10_MAX_THRESHOLD)
    df_pm10_filtered = df_pm10.dropna(subset=['PM10_AVG_filtered']).copy()
    df_pm10_filtered = df_pm10_filtered[['date', 'PM10_MIN', 'PM10_MAX_capped', 'PM10_AVG_filtered']]
    df_pm10_filtered = df_pm10_filtered.rename(columns={'PM10_AVG_filtered': 'PM10_AVG', 'PM10_MAX_capped': 'PM10_MAX'})
    df_pm10_filtered[['PM10_MIN', 'PM10_MAX', 'PM10_AVG']] = df_pm10_filtered[['PM10_MIN', 'PM10_MAX', 'PM10_AVG']].astype('float32')
    return df_pm10_filtered

# 전체 날짜 저장
"""
def save_interpolated_temperature(df):
    df.to_csv(TA_MODEL_FILE, index=False)
    saved_dates = []
    for date_value, group in df.groupby('date'):
        date_str = pd.to_datetime(date_value).strftime('%Y-%m-%d')
        save_path = os.path.join(TEMP_DIR, f'{date_str}.csv')
        if not os.path.exists(save_path):
            group.to_csv(save_path, index=False)
        saved_dates.append(date_str)
    return saved_dates

def save_processed_pm10(df):
    df.to_csv(PM10_MODEL_FILE, index=False)
    saved_dates = []
    for date_value, group in df.groupby('date'):
        date_str = pd.to_datetime(date_value).strftime('%Y-%m-%d')
        save_path = os.path.join(PM10_DIR, f'{date_str}.csv')
        if not os.path.exists(save_path):
            group.to_csv(save_path, index=False)
        saved_dates.append(date_str)
    return saved_dates
"""

# 아제 날짜만 저장
def save_interpolated_temperature(df):
    df.to_csv(TA_MODEL_FILE, index=False)
    saved_dates = []
    yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    df = df[df['date'] == yesterday]  # 어제 날짜만 필터링

    for date_value, group in df.groupby('date'):
        date_str = pd.to_datetime(date_value).strftime('%Y-%m-%d')
        save_path = os.path.join(TEMP_DIR, f'{date_str}.csv')
        group.to_csv(save_path, index=False)
        saved_dates.append(date_str)
    return saved_dates

def save_processed_pm10(df):
    df.to_csv(PM10_MODEL_FILE, index=False)
    saved_dates = []
    yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    df = df[df['date'] == yesterday]  # 어제 날짜만 필터링

    for date_value, group in df.groupby('date'):
        date_str = pd.to_datetime(date_value).strftime('%Y-%m-%d')
        save_path = os.path.join(PM10_DIR, f'{date_str}.csv')
        group.to_csv(save_path, index=False)
        saved_dates.append(date_str)
    return saved_dates

def main():
    df_ta, df_pm10 = load_data()
    df_ta_interp = interpolate_temperature(preprocess_temperature(df_ta))
    saved_temp_dates = save_interpolated_temperature(df_ta_interp)

    df_pm10_filtered = process_pm10_outliers(preprocess_pm10(df_pm10))
    saved_pm10_dates = save_processed_pm10(df_pm10_filtered)

    return saved_temp_dates, saved_pm10_dates
