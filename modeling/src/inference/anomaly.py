import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from modeling.src.utils.utils import get_output_temperature, get_output_pm10
from modeling.src.utils.aws import download_data_from_s3
from modeling.src.utils.utils import message_to_slack

def is_model_drift(project_path, model_name):
    data_root_path = os.path.join(project_path, 'data')
    anomaly_data_path = os.path.join(data_root_path, 'anomaly/inference')

    DRIFT_THRESHOLD = -1

    anomalies_files = glob.glob(os.path.join(anomaly_data_path, f"*_{model_name}_*_anomalies.csv"))
    anomalies_files.sort(key=os.path.getmtime)
    latest_anomalies_file = anomalies_files[-1]

    df = pd.read_csv(latest_anomalies_file)

    if len(df) > DRIFT_THRESHOLD:
        message_to_slack(f"이상치 탐지 결과 {len(df)} 개 발생 하여 재학습!")
    return len(df) > DRIFT_THRESHOLD

def inference(project_path, data_name, model_name):
    model_root_path = os.path.join(project_path, 'models')
    data_root_path = os.path.join(project_path, 'data')
    anomaly_data_path = os.path.join(data_root_path, 'anomaly/inference')

    os.makedirs(anomaly_data_path, exist_ok=True)
    os.makedirs(model_root_path, exist_ok=True)

    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst).strftime('%Y%m%d_%H%M%S')

    download_data_from_s3(data_root_path, data_name)
    files = glob.glob(os.path.join(data_root_path, f"*_{data_name}_data.csv"))
    files.sort(key=os.path.getmtime)
    latest_file = files[-1]

    df = pd.read_csv(latest_file)

    df_train = df[-365*5:-365].reset_index(drop=True)
    df_inference = df[-365:].reset_index(drop=True)
    df = df_inference

    if model_name == 'temperature':
        columns = get_output_temperature()
    elif model_name == 'pm10':
        columns = get_output_pm10()


    df_timestamp = df[['date']].copy()
    df = df[columns]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    class AnomalyDetectorLSTM(nn.Module):
        def __init__(self, seq_len, n_features):
            super(AnomalyDetectorLSTM, self).__init__()
            self.seq_len = seq_len
            self.n_features = n_features
            self.embedding_dim = 4

            self.encoder = nn.Sequential(
                nn.LSTM(input_size=n_features, hidden_size=16, batch_first=True),
                nn.LSTM(input_size=16, hidden_size=self.embedding_dim, batch_first=True)
            )

            self.decoder = nn.Sequential(
                nn.LSTM(input_size=self.embedding_dim, hidden_size=self.embedding_dim, batch_first=True),
                nn.LSTM(input_size=self.embedding_dim, hidden_size=16, batch_first=True),
                nn.Linear(16, n_features)
            )

        def forward(self, x):
            # Encode
            x, _ = self.encoder[0](x)
            x, (hidden, _) = self.encoder[1](x)

            # Repeat vector
            x = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)

            # Decode
            x, _ = self.decoder[0](x)
            x, _ = self.decoder[1](x)
            x = self.decoder[2](x)
            return x


    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.values)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model_path = os.path.join(model_root_path, f"lstm_{model_name}_anomaly_detector.pth")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model = AnomalyDetectorLSTM(seq_len=1, n_features=X.shape[2])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_tensor = X_tensor.to(device)
        
        X_pred = model(X_tensor).detach().cpu().numpy()


    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred_inv = scaler.inverse_transform(X_pred)
    X_pred_df = pd.DataFrame(X_pred_inv, columns=df.columns)
    X_pred_df.index = df.index

    for column in columns:
        scores = X_pred_df.copy()

        scores['date'] = pd.to_datetime(df_timestamp['date'], errors="coerce")
        scores['real'] = df[column].values
        scores['loss_mae'] = np.abs(scores['real'] - scores[column])
        scores['Threshold'] = 40
        scores['Anomaly'] = (scores['loss_mae'] > scores['Threshold']).astype(int)
        scores['anomalies'] = np.where(scores["Anomaly"] == 1, scores["real"], np.nan)

        scores = scores.sort_values("date").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(scores["date"], scores["loss_mae"], label="Loss")
        ax.plot(scores["date"], scores["Threshold"], label="Threshold", linestyle='--')

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        ax.set_title("Loss vs Threshold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(anomaly_data_path, f"{now}_{model_name}_{column}_Threshold.png"))
        plt.close()

        cols = ['date'] + [col for col in scores.columns if col != 'date']
        scores = scores[cols]
        scores[scores["Anomaly"] == 1].to_csv(
            os.path.join(anomaly_data_path, f'{now}_{model_name}_{column}_anomalies.csv'),
            index=False
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(scores["date"], scores["real"], label=column)
        if scores["Anomaly"].sum() > 0:
            mask = scores["Anomaly"] == 1
            ax.scatter(scores.loc[mask, "date"], scores.loc[mask, "anomalies"],
                    color="red", label="Anomaly", s=25)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        ax.set_title("Anomalies Detected (Inference)")
        ax.set_xlabel("Date")
        ax.set_ylabel(column)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(anomaly_data_path, f"{now}_{model_name}_{column}_Anomaly.png"))
        plt.close()