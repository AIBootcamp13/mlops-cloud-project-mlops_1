import os
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

def train(project_path, data_name, model_name):
    model_root_path = os.path.join(project_path, 'models')
    data_root_path = os.path.join(project_path, 'data')
    anomaly_data_path = os.path.join(data_root_path, 'anomaly/train')

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
    df = df_train

    if model_name == 'temperature':
        columns = get_output_temperature()
    elif model_name == 'pm10':
        columns = get_output_pm10()

    df_timestamp = df[['date']].copy()
    df = df[columns]

    train_prp = 0.6
    train = df.iloc[:int(df.shape[0]*train_prp)]
    test = df.iloc[int(df.shape[0]*train_prp):]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train)
    X_test = scaler.transform(test)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print(f"train shape : {X_train.shape}, test shape : {X_test.shape}")

    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42, shuffle=True)

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

    epochs = 10
    batch_size = 10
    learning_rate = 1e-3

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor), batch_size=batch_size, shuffle=False)

    model = AnomalyDetectorLSTM(seq_len=X_train.shape[1], n_features=X_train.shape[2])
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_epoch_loss = 0
        for batch in train_loader:
            batch = batch[0]
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        
        avg_train_loss = train_epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0]
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_epoch_loss += loss.item()
        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


    model_path = os.path.join(model_root_path, f"lstm_{model_name}_anomaly_detector.pth")
    torch.save(model.state_dict(), model_path)


    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Autoencoder MAE Loss over Epochs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(anomaly_data_path, f"{now}_{model_name}_Error_Loss.png"))


    with torch.no_grad():
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_tensor = X_tensor.to(device)
        X_pred = model(X_tensor).detach().cpu().numpy()

    X_pred = X_pred[:, -1, :]
    X_pred_inv = scaler.inverse_transform(X_pred)
    pred_df = pd.DataFrame(X_pred_inv, columns=df.columns)

    scores = pd.DataFrame()

    for column in columns:
        scores[f"{column}_train"] = df[column].values[:len(pred_df)]
        scores[f"{column}_predicted"] = pred_df[column]
        scores[f"{column}_loss_mae"] = (scores[f"{column}_train"] - scores[f"{column}_predicted"]).abs()


    for column in columns:
        plt.figure(figsize=(8, 4))
        plt.hist(scores[f"{column}_loss_mae"], bins=50)
        plt.title(f"{model_name} {column} Error Distribution (Train)")
        plt.xlabel(f"MAE {column}")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(anomaly_data_path, f"{now}_{model_name}_{column}_Error_Distribution.png"))

    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        X_tensor = X_tensor.to(device)
        X_pred = model(X_tensor).detach().cpu().numpy()

    X_pred = X_pred[:, -1, :]
    X_pred_inv = scaler.inverse_transform(X_pred)
    X_pred_df = pd.DataFrame(X_pred_inv, columns=df.columns)
    X_pred_df.index = test.index

    for column in columns:
        scores = X_pred_df.copy()

        scores['date'] = pd.to_datetime(df_timestamp.loc[scores.index, 'date'], errors='coerce')
        scores['real'] = test[column].values
        scores['loss_mae'] = np.abs(scores['real'] - scores[column])
        scores['Threshold'] = 10
        scores['Anomaly'] = (scores['loss_mae'] > scores['Threshold']).astype(int)
        scores['anomalies'] = np.where(scores['Anomaly'] == 1, scores['real'], np.nan)

        scores = scores.sort_values(by='date').reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(scores['date'], scores['loss_mae'], label='Loss')
        ax.plot(scores['date'], scores['Threshold'], label='Threshold', linestyle='--')

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
            os.path.join(anomaly_data_path, f"{now}_{model_name}_{column}_anomalies.csv"),
            index=False
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(scores['date'], scores['real'], label=column)

        if scores['Anomaly'].sum() > 0:
            mask = scores['Anomaly'] == 1
            ax.scatter(scores.loc[mask, 'date'], scores.loc[mask, 'anomalies'],
                    color='red', label='Anomaly', s=25)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        ax.set_title("Anomalies Detected (Test)")
        ax.set_xlabel("Date")
        ax.set_ylabel(column)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(anomaly_data_path, f"{now}_{model_name}_{column}_Anomaly.png"))
        plt.close()
