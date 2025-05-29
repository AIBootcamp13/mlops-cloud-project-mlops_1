import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import boto3
from botocore.exceptions import NoCredentialsError
from modeling.src.model.lstm import MultiOutputLSTM
    
def model_save(model, model_root_path, model_name):
    model_path = os.path.join(model_root_path, f"lstm_{model_name}.pth")
    torch.save(model.state_dict(), model_path)

    try:
        s3 = boto3.client('s3')
        s3.list_buckets()
        print("Connect S3 Successes")
        s3.upload_file(model_path, "mlops-study-web-lmw", f"models/model_{model_name}_v1.pth")
    except NoCredentialsError:
        print("Failed S3 Successes")

def train(model_root_path, data, outputs, scaler, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = data

    WINDOW_SIZE = 30

    features = df[outputs].values
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    def create_lstm_sequences(values, window_size=7):
        X, y = [], []
        for i in range(len(values) - window_size):
            X.append(values[i:i+window_size])
            y.append(values[i+window_size])
        return np.array(X), np.array(y)

    X, y = create_lstm_sequences(features_scaled, window_size=WINDOW_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2025)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    print(f"X_train shape: {X_train_tensor.shape}, y_train shape: {y_train_tensor.shape}")

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)

    model = MultiOutputLSTM(outputs).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []
    epochs = 1

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                loss = criterion(model(Xb), yb)
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"[{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    model_save(model, model_root_path, model_name)