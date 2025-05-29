import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


from dotenv import load_dotenv

import pandas as pd
import numpy as np
import torch
import fire

from modeling.src.inference.inference import (
    init_model, inference, temperature_to_df, PM_to_df, get_scalers, get_outputs
)
from modeling.src.train.train import train
from modeling.src.postprocess.postprocess import write_db, read_db

WINDOW_SIZE = 30

def run_train_temperature(data_root_path, model_root_path, outputs_temperature, scaler_temperature):
    data = pd.read_csv(os.path.join(data_root_path, 'TA_data.csv'))

    return train(model_root_path, data, outputs_temperature, scaler_temperature, "temperature")

def run_train_PM(data_root_path, model_root_path, outputs_PM, scaler_PM):
    data = pd.read_csv(os.path.join(data_root_path, 'PM10_data.csv'))

    return train(model_root_path, data, outputs_PM, scaler_PM, "PM")

def run_train(data_root_path, model_root_path):

    scaler_temperature, scaler_PM = get_scalers(data_root_path)
    outputs_temperature, outputs_PM = get_outputs()

    val_loss_temperature = run_train_temperature(data_root_path, model_root_path, outputs_temperature, scaler_temperature)
    val_loss_PM = run_train_PM(data_root_path, model_root_path, outputs_PM, scaler_PM)
    return f'total val_loss temperature : {val_loss_temperature}, PM : {val_loss_PM}'

def run_inference_temperature(data_root_path, model_root_path, model, scaler, outputs, device):
    fake_test_data = np.random.normal(loc=15, scale=3, size=(WINDOW_SIZE, len(outputs)))

    results = inference(model, fake_test_data, scaler, outputs, device)    
    # temperature_df = temperature_to_df(results, outputs)
    # print(temperature_df)

    # write_db(temperature_df, "mlops", "temperature")
    return results

def run_inference_PM(data_root_path, model_root_path, model, scaler, outputs, device):
    fake_test_data = np.random.normal(loc=15, scale=3, size=(WINDOW_SIZE, len(outputs)))

    results = inference(model, fake_test_data, scaler, outputs, device)    
    # PM_df = PM_to_df(results, outputs)
    # print(PM_df)

    # write_db(PM_df, "mlops", "PM")
    return results

def run_inference(data_root_path, model_root_path, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    model_temperature, model_PM = init_model(model_root_path)
    scaler_temperature, scaler_PM = get_scalers(data_root_path)
    outputs_temperature, outputs_PM = get_outputs()

    temperature_results = run_inference_temperature(data_root_path, model_root_path, model_temperature, scaler_temperature, outputs_temperature, device)
    PM_results = run_inference_PM(data_root_path, model_root_path, model_PM, scaler_PM, outputs_PM, device)
    return temperature_results, PM_results

def main(run_mode, data_root_path, model_root_path):
    load_dotenv()

    if run_mode == "train":
        val_loss = run_train(data_root_path, model_root_path)
        print(val_loss)
    elif run_mode == "inference":
        temperature_results, PM_results = run_inference(data_root_path, model_root_path)
        print(temperature_results, PM_results)

if __name__ == '__main__':
    fire.Fire(main)

    