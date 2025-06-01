import sys
import os

from dotenv import load_dotenv
load_dotenv()

import mlflow

from modeling.src.train.train import train_airflow, model_save_airflow


mlflow_url = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(mlflow_url)
mlflow.set_experiment("WeatherExperiment")

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run():
    params = {
        "filename": "feature_data.csv",
        "sequence_lenght": 30, 
        "epcohs": 40,
        "hidden_size": 128,
        "num_layers": 3,
        "lr": 0.01,
        # "save_path": model_path
    }

    with mlflow.start_run():
        mlflow.log_params(params)

        mlflow.pytorch.log_model()
        # mlflow.log_artifact(model_saved_path)
        mlflow.log_metric("val loss", 0)

if __name__ == "__main__":
    run()