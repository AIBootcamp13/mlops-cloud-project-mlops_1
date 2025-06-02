import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_outputs():
    outputs_temperature = ["TA_AVG", "TA_MAX", "TA_MIN"]
    outputs_PM = ["PM10_MIN", "PM10_MAX", "PM10_AVG"]
    return outputs_temperature, outputs_PM

def get_scaler(data_path, outputs):
    df = pd.read_csv(data_path)
    features = df[outputs].values
    scaler = MinMaxScaler()
    scaler.fit_transform(features)
    return scaler

def temperature_to_df(results, outputs):
    return pd.DataFrame(
        data=[[results[outputs[0]], results[outputs[1]], results[outputs[2]]]],
        columns=outputs
    )

def PM_to_df(results, outputs):
    return pd.DataFrame(
        data=[[results[outputs[0]], results[outputs[1]], results[outputs[2]]]],
        columns=outputs
    )

CFG = {
    'WINDOW_SIZE': 7,
    'EPOCHS': 30,
}