import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_scaler_temperature(data_root_path, outputs):
    df = pd.read_csv(os.path.join(data_root_path, 'TA_data.csv'))
    features = df[outputs].values
    scaler = MinMaxScaler()
    scaler.fit_transform(features)
    return scaler

def get_scaler_PM(data_root_path, outputs):
    df = pd.read_csv(os.path.join(data_root_path, 'PM10_data.csv'))
    features = df[outputs].values
    scaler = MinMaxScaler()
    scaler.fit_transform(features)
    return scaler

def get_outputs():
    outputs_temperature = ["TA_AVG", "TA_MAX", "TA_MIN"]
    outputs_PM = ["PM10_MIN", "PM10_MAX", "PM10_AVG"]
    return outputs_temperature, outputs_PM

def get_scalers(data_root_path):
    outputs_temperature, outputs_PM = get_outputs()
    return get_scaler_temperature(data_root_path, outputs_temperature), get_scaler_PM(data_root_path, outputs_PM)

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
    'IMG_SIZE': 224,
    'BATCH_SIZE': 64,
    'EPOCHS': 30,
    'LEARNING_RATE': 1e-4,
    'SEED': 42,

    'EXPERIMENT_NAME' : None,
    'WRONG_DIR' : None
}