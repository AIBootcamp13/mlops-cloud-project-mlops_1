import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import date
from config import TA_PROCESSED_FILE, TA_MODEL_FILE, PM10_PROCESSED_FILE, PM10_MODEL_FILE

def load_data():
    df_ta = pd.read_csv(TA_PROCESSED_FILE, parse_dates=['date'])
    df_pm10 = pd.read_csv(PM10_PROCESSED_FILE, parse_dates=['date'])
    return df_ta, df_pm10

def preprocess_temperature(df_ta):
    today = date.today()
    df_ta = df_ta[df_ta['date'] < pd.to_datetime(today)]
    df_ta = df_ta.replace(-99.0, np.nan)

    return df_ta

def interpolate_temperature(df_ta):
    df_ta_interpolated = df_ta.set_index('date').interpolate(method='time').reset_index()
    df_ta_interpolated = df_ta_interpolated.dropna()
    
    return df_ta_interpolated

def preprocess_pm10(df_pm10):
    today = date.today()
    df_pm10 = df_pm10[df_pm10['date'] < pd.to_datetime(today)]
    
    return df_pm10

def process_pm10_outliers(df_pm10):
    df_pm10['PM10_AVG_filtered'] = df_pm10['PM10_AVG'].where(df_pm10['PM10_AVG'] <= 90.8)
    df_pm10['PM10_MAX_capped'] = df_pm10['PM10_MAX'].clip(upper=160.5)
    df_pm10_filtered = df_pm10.dropna(subset=['PM10_AVG_filtered']).copy()
    df_pm10_filtered = df_pm10_filtered[['date', 'PM10_MIN', 'PM10_MAX_capped', 'PM10_AVG_filtered']]
    df_pm10_filtered = df_pm10_filtered.rename(columns={
        'PM10_AVG_filtered': 'PM10_AVG',
        'PM10_MAX_capped': 'PM10_MAX'
    })
    
    return df_pm10_filtered

def save_processed_temperature(df_ta):
    df_ta.to_csv(TA_PROCESSED_FILE, index=False)
    
def save_interpolated_temperature(df_ta_interpolated):
    df_ta_interpolated.to_csv(TA_MODEL_FILE, index=False)

def save_processed_pm10(df_pm10_filtered):
    df_pm10_filtered.to_csv(PM10_MODEL_FILE, index=False)
    
def main():
    df_ta, df_pm10 = load_data()
    
    df_ta_cleaned = preprocess_temperature(df_ta)
    df_ta_interpolated = interpolate_temperature(df_ta_cleaned)
    save_interpolated_temperature(df_ta_interpolated)

    df_pm10 = preprocess_pm10(df_pm10)
    df_pm10_filtered = process_pm10_outliers(df_pm10)
    save_processed_pm10(df_pm10_filtered)