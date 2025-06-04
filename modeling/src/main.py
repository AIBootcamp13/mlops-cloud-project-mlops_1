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
import mlflow

from modeling.src.train.anomaly import train
from modeling.src.train.train import run_train
from modeling.src.inference.inference import run_inference
from modeling.src.mlflow.mlflow import run_mlflow_tester

from modeling.src.utils.utils import project_path

def main(run_mode, batch_size=64):
    load_dotenv()

    data_root_path = os.path.join(project_path(), 'data')
    model_root_path = os.path.join(project_path(), 'models')

    if run_mode == "train":
        val_loss = run_train(data_root_path, model_root_path, batch_size)
        print(val_loss)
    elif run_mode == "inference":
        temperature_results, PM_results = run_inference(data_root_path, model_root_path, batch_size)
        print(temperature_results, PM_results)
    elif run_mode == "mlflow-tester":
        mlflow_url = os.getenv("MLFLOW_HOST")
        mlflow.set_tracking_uri(mlflow_url)
        mlflow.set_experiment("WeatherExperiment")
        run_mlflow_tester()
    elif run_mode == "anomaly-train":
        train(project_path(), 'temperature', 'temperature')

if __name__ == '__main__':
    fire.Fire(main)
