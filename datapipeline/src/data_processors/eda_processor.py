import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from src.config.settings import (
    TA_PROCESSED_FILE, TA_MODEL_FILE, PM10_PROCESSED_FILE, PM10_MODEL_FILE
)

PM10_AVG_THRESHOLD = 90.8
PM10_MAX_THRESHOLD = 160.5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'temperature')
PM10_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'pm10')
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PM10_DIR, exist_ok=True)

def load_data():
    df_ta = pd.read_csv(TA_PROCESSED_FILE, parse_dates=['date'])
    df_pm10 = pd.read_csv(PM10_PROCESSED_FILE, parse_dates=['date'])
    return df_ta, df_pm10

def preprocess_temperature(df_ta, days=None, reference_date=None):
    if reference_date is None:
        reference_date = pd.to_datetime(date.today())
    if days:
        start_date = reference_date - timedelta(days=days)
        df_ta = df_ta[(df_ta['date'] < reference_date) & (df_ta['date'] >= start_date)]
    df_ta = df_ta.replace(-99.0, np.nan)
    df_ta[['TA_AVG', 'TA_MAX', 'TA_MIN']] = df_ta[['TA_AVG', 'TA_MAX', 'TA_MIN']].astype('float32')
    return df_ta

def interpolate_temperature(df_ta):
    return df_ta.set_index('date').interpolate(method='time', limit_direction='both').dropna().reset_index()

def preprocess_pm10(df_pm10, days=None, reference_date=None):
    if reference_date is None:
        reference_date = pd.to_datetime(date.today())
    if days:
        start_date = reference_date - timedelta(days=days)
        df_pm10 = df_pm10[(df_pm10['date'] < reference_date) & (df_pm10['date'] >= start_date)]
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
    return df_pm10_filtered

def save_if_changed(df, path):
    if os.path.exists(path):
        old_df = pd.read_csv(path)
        if df.equals(old_df):
            return False
    df.to_csv(path, index=False)
    return True

def save_interpolated_temperature(df):
    save_if_changed(df, TA_MODEL_FILE)
    saved_dates = []
    for date_value, group in df.groupby('date'):
        date_str = pd.to_datetime(date_value).strftime('%Y-%m-%d')
        save_path = os.path.join(TEMP_DIR, f'{date_str}.csv')
        if save_if_changed(group, save_path):
            saved_dates.append(date_str)
    return saved_dates

def save_processed_pm10(df):
    save_if_changed(df, PM10_MODEL_FILE)
    saved_dates = []
    for date_value, group in df.groupby('date'):
        date_str = pd.to_datetime(date_value).strftime('%Y-%m-%d')
        save_path = os.path.join(PM10_DIR, f'{date_str}.csv')
        if save_if_changed(group, save_path):
            saved_dates.append(date_str)
    return saved_dates

def run_eda_for_recent_days(days=14, reference_date=None):
    if reference_date is None:
        reference_date = pd.to_datetime(date.today())
    df_ta, df_pm10 = load_data()
    df_ta_interp = interpolate_temperature(preprocess_temperature(df_ta, days, reference_date))
    df_pm10_filtered = process_pm10_outliers(preprocess_pm10(df_pm10, days, reference_date))
    saved_temp_dates = save_interpolated_temperature(df_ta_interp)
    saved_pm10_dates = save_processed_pm10(df_pm10_filtered)
    return saved_temp_dates, saved_pm10_dates

def run_full_eda():
    return run_eda_for_recent_days(days=None)
