import os

import torch

from modeling.src.model.lstm import MultiOutputLSTM
from modeling.src.utils.utils import get_outputs

def init_model(model_root_path):
    outputs_temperature, outputs_PM = get_outputs()
    
    model_temperature_path = os.path.join(model_root_path, "lstm_temperature.pth")

    model_temperature_checkpoint = torch.load(model_temperature_path, map_location=torch.device('cpu'), weights_only=True)

    model_temperature = MultiOutputLSTM(outputs_temperature)
    model_temperature.load_state_dict(model_temperature_checkpoint)

    model_PM_path = os.path.join(model_root_path, "lstm_PM.pth")

    model_PM_checkpoint = torch.load(model_PM_path, map_location=torch.device('cpu'), weights_only=True)

    model_PM = MultiOutputLSTM(outputs_PM)
    model_PM.load_state_dict(model_PM_checkpoint)

    return model_temperature, model_PM

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
