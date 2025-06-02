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

from modeling.src.train.train import run_temperature_train, run_pm_train
from modeling.src.inference.inference import run_inference_PM, run_inference_temperature

def run_train(data_root_path, model_root_path, batch_size):
    
    _, val_loss_temperature = run_temperature_train(data_root_path, model_root_path, batch_size=batch_size)
    _, val_loss_PM = run_pm_train(data_root_path, model_root_path, batch_size=batch_size)

    return f'total val_loss temperature : {val_loss_temperature}, PM : {val_loss_PM}'

def run_inference(data_root_path, model_root_path, batch_size):
    
    temperature_results = run_inference_temperature(data_root_path, model_root_path, batch_size=batch_size)
    PM_results = run_inference_PM(data_root_path, model_root_path, batch_size=batch_size)
    return temperature_results, PM_results

def main(run_mode, data_root_path, model_root_path, batch_size=64):
    load_dotenv()

    if run_mode == "train":
        val_loss = run_train(data_root_path, model_root_path, batch_size)
        print(val_loss)
    elif run_mode == "inference":
        temperature_results, PM_results = run_inference(data_root_path, model_root_path, batch_size)
        print(temperature_results, PM_results)

if __name__ == '__main__':
    fire.Fire(main)

    