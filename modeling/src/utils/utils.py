import io
import os
import sys
import glob

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests
import json


def get_output_temperature():
    return ["TA_AVG", "TA_MAX", "TA_MIN"]

def get_output_pm10():
    return ["PM10_MIN", "PM10_MAX", "PM10_AVG"]

def get_outputs():
    outputs_temperature = get_output_temperature()
    outputs_PM = get_output_pm10()
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

def message_to_slack(message):
    url = os.getenv('SLACK_WEBHOOK_URL')
    if not url:
        raise ValueError("SLACK_WEBHOOK_URL environment variable not set")

    data = {"text": message}
    headers = {'Content-type': 'application/json'}

    requests.post(url, data=json.dumps(data), headers=headers)

def project_path():
    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
        "..",
        ".."
    )

CFG = {
    'WINDOW_SIZE': 7,
    'EPOCHS': 5,
}