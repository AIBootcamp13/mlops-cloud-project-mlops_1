import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.data_loaders.temp_loader import run_temp_preprocessing
from src.data_loaders.pm10_loader import run_pm10_preprocessing

PM10_AVG_THRESHOLD = 90.8
PM10_MAX_THRESHOLD = 160.5

def preprocess_temperature(df_ta, days=None, reference_date=None):
    if reference_date is None:
        # 어제 날짜를 기준으로 설정 (데이터 로더와 일관성 유지)
        reference_date = pd.to_datetime(date.today() - timedelta(days=1))
    if days:
        start_date = reference_date - timedelta(days=days)
        df_ta = df_ta[(df_ta['date'] <= reference_date) & (df_ta['date'] >= start_date)]
    df_ta = df_ta.replace(-99.0, np.nan)
    df_ta[['TA_AVG', 'TA_MAX', 'TA_MIN']] = df_ta[['TA_AVG', 'TA_MAX', 'TA_MIN']].astype('float32')
    df_ta[['TA_AVG', 'TA_MAX', 'TA_MIN']] = df_ta[['TA_AVG', 'TA_MAX', 'TA_MIN']].round(1)
    return df_ta

def interpolate_temperature(df_ta):
    return df_ta.set_index('date').interpolate(method='time', limit_direction='both').dropna().reset_index()

def preprocess_pm10(df_pm10, days=None, reference_date=None):
    if reference_date is None:
        # 어제 날짜를 기준으로 설정 (데이터 로더와 일관성 유지)
        reference_date = pd.to_datetime(date.today() - timedelta(days=1))
    if days:
        start_date = reference_date - timedelta(days=days)
        df_pm10 = df_pm10[(df_pm10['date'] <= reference_date) & (df_pm10['date'] >= start_date)]
    return df_pm10

def process_pm10_outliers(df_pm10):
    df_pm10['PM10_AVG_filtered'] = df_pm10['PM10_AVG'].where(df_pm10['PM10_AVG'] <= PM10_AVG_THRESHOLD)
    df_pm10['PM10_MAX_capped'] = df_pm10['PM10_MAX'].clip(upper=PM10_MAX_THRESHOLD)
    df_pm10_filtered = df_pm10.dropna(subset=['PM10_AVG_filtered']).copy()
    df_pm10_filtered = df_pm10_filtered[['date', 'PM10_MIN', 'PM10_MAX_capped', 'PM10_AVG_filtered']]
    df_pm10_filtered = df_pm10_filtered.rename(columns={
        'PM10_AVG_filtered': 'PM10_AVG',
        'PM10_MAX_capped': 'PM10_MAX'
    })
    df_pm10_filtered[['PM10_MIN', 'PM10_MAX', 'PM10_AVG']] = df_pm10_filtered[
        ['PM10_MIN', 'PM10_MAX', 'PM10_AVG']
    ].astype('float32')
    df_pm10_filtered[['PM10_MIN', 'PM10_MAX', 'PM10_AVG']] = df_pm10_filtered[['PM10_MIN', 'PM10_MAX', 'PM10_AVG']].round(1)
    return df_pm10_filtered

def run_eda_for_recent_days(df_ta, df_pm10, days=14, reference_date=None):
    if reference_date is None:
        # 어제 날짜를 기준으로 설정 (데이터 로더와 일관성 유지)
        reference_date = pd.to_datetime(date.today() - timedelta(days=1))
    df_ta_interp = interpolate_temperature(preprocess_temperature(df_ta, days, reference_date))
    df_pm10_filtered = process_pm10_outliers(preprocess_pm10(df_pm10, days, reference_date))
    return df_ta_interp, df_pm10_filtered

def run_full_eda(df_ta, df_pm10):
    return run_eda_for_recent_days(df_ta, df_pm10, days=None)

def run_eda_for_recent_days_with_fetch(days=14, reference_date=None):
    df_ta = run_temp_preprocessing()
    df_pm10 = run_pm10_preprocessing()
    return run_eda_for_recent_days(df_ta, df_pm10, days, reference_date)