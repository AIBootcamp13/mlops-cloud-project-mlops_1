import os
import glob

import pandas as pd

from modeling.src.trainer.baselineTrainer import BaselineTrainer
from modeling.src.utils.utils import get_outputs, get_scaler
from modeling.src.utils.utils import CFG

def run_temperature_train(data_root_path, model_root_path, batch_size, model_name="MULTI_OUTPUT_LSTM"):
    epochs = CFG["EPOCHS"]
    window_size = CFG["WINDOW_SIZE"]

    data_path = os.path.join(data_root_path, 'TA_data.csv')
    outputs_temperature, outputs_PM = get_outputs()
    scaler = get_scaler(data_path, outputs_temperature)

    data = pd.read_csv(data_path)

    save_model_name = "temperature"

    trainer = BaselineTrainer(model_name, epochs, batch_size, outputs_temperature, scaler, window_size)
    trainer.split_data(data)
    model, val_loss = trainer.train_model()
    trainer.save_model(model_root_path, save_model_name, False)

    return model, val_loss

def run_pm_train(data_root_path, model_root_path, batch_size, model_name="MULTI_OUTPUT_LSTM"):
    epochs = CFG["EPOCHS"]
    window_size = CFG["WINDOW_SIZE"]

    files = glob.glob(os.path.join(data_root_path, "*_PM10_data.csv"))
    files.sort(key=os.path.getmtime)
    latest_anomalies_file = files[-1]

    data_path = os.path.join(data_root_path, latest_anomalies_file)
    outputs_temperature, outputs_PM = get_outputs()
    scaler = get_scaler(data_path, outputs_PM)

    data = pd.read_csv(data_path)

    save_model_name = "PM"

    trainer = BaselineTrainer(model_name, epochs, batch_size, outputs_PM, scaler, window_size)
    trainer.split_data(data)
    model, val_loss = trainer.train_model()
    trainer.save_model(model_root_path, save_model_name, False)

    return model, val_loss