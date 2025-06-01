import os
import shutil
import sys

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(project_path)
sys.path.append(project_path)

from dotenv import load_dotenv
load_dotenv()

import torch
import mlflow
from mlflow.models.signature import infer_signature

from modeling.src.train.train import run_temperature_train, run_pm_train


mlflow_url = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(mlflow_url)
mlflow.set_experiment("WeatherExperiment")

def run_mlflow(run_name, model_name):
    params = {
        "model_name": model_name
    }

    input_example = torch.randn(64, 30, 3)
    output_example = torch.randn(64, 30)

    signature = infer_signature(
        input_example.numpy(),
        output_example.numpy()
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        data_root_path = os.path.join(project_path, 'data')
        model_root_path = os.path.join(project_path, 'models')
        
        if run_name == "temperature":
            model, val_loss = run_temperature_train(data_root_path, model_root_path, **params)
        elif run_name == "PM":
            model, val_loss = run_pm_train(data_root_path, model_root_path, **params)
        
        mlflow.pytorch.log_model(model, 'models', input_example=input_example.numpy(), signature=signature)
        mlflow.log_artifacts(model_root_path, artifact_path='checkpoints')
        mlflow.log_metric("val loss", val_loss)

def main():
    run_names = ["temperature", "PM"]
    model_names = ["MULTI_OUTPUT_LSTM", "MULTI_OUTPUT_STACKED_LSTM"]
    for run_name in run_names:
        for model_name in model_names:
            run_mlflow(run_name, model_name)

if __name__ == "__main__":
    main()