import os
import shutil
import sys

import torch
import mlflow
from mlflow.models.signature import infer_signature

from modeling.src.train.train import run_temperature_train, run_pm10_train
from modeling.src.utils.utils import project_path

def run_experiments(run_name, model_name, batch_size):
    params = {
        "model_name": model_name,
        "batch_size": batch_size
    }

    input_example = torch.randn(64, 30, 3)
    output_example = torch.randn(64, 30)

    signature = infer_signature(
        input_example.numpy(),
        output_example.numpy()
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        data_root_path = os.path.join(project_path(), 'data')
        model_root_path = os.path.join(project_path(), 'models')
        
        if run_name == "temperature":
            model, val_loss = run_temperature_train(data_root_path, model_root_path, **params)
        elif run_name == "PM":
            model, val_loss = run_pm10_train(data_root_path, model_root_path, **params)
        
        mlflow.pytorch.log_model(model, 'models', input_example=input_example.numpy(), signature=signature)
        mlflow.log_artifacts(model_root_path, artifact_path='checkpoints')
        mlflow.log_metric("val loss", val_loss)

def run_mlflow_tester():
    run_names = ["temperature", "PM"]
    model_names = ["MULTI_OUTPUT_LSTM", "MULTI_OUTPUT_STACKED_LSTM"]
    batch_sizes = [4, 8, 16, 32, 64]

    for run_name in run_names:
        for model_name in model_names:
            for batch_size in batch_sizes:
                run_experiments(run_name, model_name, batch_size)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    mlflow_url = os.getenv("MLFLOW_HOST")
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment("WeatherExperiment")

    run_mlflow_tester()