import os

import torch
import pandas as pd
import numpy as np

from modeling.src.model.lstm import MultiOutputLSTM
from modeling.src.utils.utils import get_outputs, get_scaler
from modeling.src.utils.utils import CFG
from modeling.src.utils.constant import Models

def init_model(model_path, model_name, outputs):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

    Models.validation(model_name)
    model_class = Models[model_name.upper()].value
    
    model = model_class(outputs)
    model.load_state_dict(checkpoint)

    return model

def inference(model, data, scaler, outputs, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_scaled = scaler.transform(data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input_tensor)
        prediction = output.cpu().numpy().squeeze()
        result = scaler.inverse_transform([prediction])
    return {
        outputs[0]: result[0][0], 
        outputs[1]: result[0][1], 
        outputs[2]: result[0][2]}


def run_inference_temperature(data_root_path, model_root_path, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    window_size = CFG['WINDOW_SIZE']

    data_path = os.path.join(data_root_path, 'TA_data.csv')
    model_path = os.path.join(model_root_path, 'lstm_temperature.pth')

    outputs_temperature, outputs_PM = get_outputs()
    scaler = get_scaler(data_path, outputs_temperature)
    
    model = init_model(model_path, 'multi_output_lstm', outputs_temperature)

    fake_test_data = np.random.normal(loc=15, scale=3, size=(window_size, len(outputs_temperature)))

    results = inference(model, fake_test_data, scaler, outputs_temperature, device)    
    # temperature_df = temperature_to_df(results, outputs)
    # print(temperature_df)

    # write_db(temperature_df, "mlops", "temperature")
    return results

def run_inference_PM(data_root_path, model_root_path, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    window_size = CFG['WINDOW_SIZE']

    data_path = os.path.join(data_root_path, 'PM10_data.csv')
    model_path = os.path.join(model_root_path, 'lstm_PM.pth')

    outputs_temperature, outputs_PM = get_outputs()
    scaler = get_scaler(data_path, outputs_PM)
    
    model = init_model(model_path, 'multi_output_lstm', outputs_PM)

    fake_test_data = np.random.normal(loc=15, scale=3, size=(window_size, len(outputs_PM)))

    results = inference(model, fake_test_data, scaler, outputs_PM, device)    
    # temperature_df = temperature_to_df(results, outputs)
    # print(temperature_df)

    # write_db(temperature_df, "mlops", "temperature")
    return results
