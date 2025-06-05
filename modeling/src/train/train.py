import os
import glob

import pandas as pd

from modeling.src.trainer.baselineTrainer import BaselineTrainer
from modeling.src.utils.utils import get_outputs, get_scaler, message_to_slack
from modeling.src.utils.utils import CFG
from modeling.src.utils.aws import download_data_from_s3

def run_temperature_train(data_root_path, model_root_path, batch_size, model_name="MULTI_OUTPUT_LSTM"):
    epochs = CFG["EPOCHS"]
    window_size = CFG["WINDOW_SIZE"]
    
    data_name = 'temperature'
    download_data_from_s3(data_root_path, data_name)

    files = glob.glob(os.path.join(data_root_path, f"*_{data_name}_data.csv"))
    files.sort(key=os.path.getmtime)
    latest_file = files[-1]

    data_path = os.path.join(data_root_path, latest_file)
    outputs_temperature, outputs_PM = get_outputs()
    scaler = get_scaler(data_path, outputs_temperature)

    data = pd.read_csv(data_path)

    save_model_name = "temperature"

    trainer = BaselineTrainer(model_name, epochs, batch_size, outputs_temperature, scaler, window_size)
    trainer.split_data(data)
    model, val_loss = trainer.train_model()
    trainer.save_model(model_root_path, save_model_name, False)

    return model, val_loss

def run_pm10_train(data_root_path, model_root_path, batch_size, model_name="MULTI_OUTPUT_LSTM"):
    epochs = CFG["EPOCHS"]
    window_size = CFG["WINDOW_SIZE"]

    data_name = 'pm10'
    download_data_from_s3(data_root_path, data_name)

    files = glob.glob(os.path.join(data_root_path, f"*_{data_name}_data.csv"))
    files.sort(key=os.path.getmtime)
    latest_file = files[-1]

    data_path = os.path.join(data_root_path, latest_file)
    outputs_temperature, outputs_PM = get_outputs()
    scaler = get_scaler(data_path, outputs_PM)

    data = pd.read_csv(data_path)

    save_model_name = "pm10"

    trainer = BaselineTrainer(model_name, epochs, batch_size, outputs_PM, scaler, window_size)
    trainer.split_data(data)
    model, val_loss = trainer.train_model()
    trainer.save_model(model_root_path, save_model_name, False)

    return model, val_loss

def run_temperature_train_on_airflow(data_root_path, model_root_path, batch_size, model_name="MULTI_OUTPUT_LSTM"):
    _, val_loss = run_temperature_train(data_root_path, model_root_path, batch_size, model_name)
    message_to_slack(f"temperature val loss : {val_loss}")

def run_pm10_train_on_airflow(data_root_path, model_root_path, batch_size, model_name="MULTI_OUTPUT_LSTM"):
    _, val_loss = run_pm10_train(data_root_path, model_root_path, batch_size, model_name)
    message_to_slack(f"pm10 val loss : {val_loss}")

def run_train(data_root_path, model_root_path, batch_size):
    
    _, val_loss_temperature = run_temperature_train(data_root_path, model_root_path, batch_size=batch_size)
    _, val_loss_PM = run_pm10_train(data_root_path, model_root_path, batch_size=batch_size)

    return f'total val_loss temperature : {val_loss_temperature}, PM : {val_loss_PM}'
