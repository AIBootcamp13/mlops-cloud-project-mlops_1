import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import boto3
from botocore.exceptions import NoCredentialsError
from modeling.src.utils.constant import Models
from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(self, model_name, epochs, batch_size, outputs, scaler, window_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.outputs = outputs
        self.scaler = scaler
        self.window_size = window_size

        print(model_name)

        Models.validation(model_name)
        model_class = Models[model_name.upper()].value
        self.model = model_class(outputs)

        self.data = None

    @abstractmethod
    def split_data(self, data):
        pass

    @abstractmethod
    def train_model(self):
        pass
    
    @abstractmethod
    def save_model(self, model_root_path, save_model_name, is_store_in_s3=False):
        pass

    def split_data_common(self, data):
        df = data

        WINDOW_SIZE = self.window_size

        features = df[self.outputs].values
        features_scaled = self.scaler.fit_transform(features)

        def create_lstm_sequences(values, window_size=7):
            X, y = [], []
            for i in range(len(values) - window_size):
                X.append(values[i:i+window_size])
                y.append(values[i+window_size])
            return np.array(X), np.array(y)
        
        X, y = create_lstm_sequences(features_scaled, window_size=WINDOW_SIZE)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2025)
        self.data = {'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val}

    def train_model_common(self):
        X_train, X_val, y_train, y_val = self.data['X_train'], self.data['X_val'], self.data['y_train'], self.data['y_val']

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        print(f"X_train shape: {X_train_tensor.shape}, y_train shape: {y_train_tensor.shape}")

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=self.batch_size)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        train_losses, val_losses = [], []

        self.model.to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(Xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * Xb.size(0)
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    loss = criterion(self.model(Xb), yb)
                    val_loss += loss.item() * Xb.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            print(f"[{epoch+1}/{self.epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return self.model, val_loss
    
    def save_model_common(self, model_root_path, save_model_name, is_store_in_s3=False):
        os.makedirs(model_root_path, exist_ok=True)

        model_path = os.path.join(model_root_path, f"lstm_{save_model_name}.pth")
        torch.save(self.model.state_dict(), model_path)

        if is_store_in_s3 == True:
            try:
                s3 = boto3.client('s3')
                s3.list_buckets()
                print("Connect S3 Successes")
                s3.upload_file(model_path, "mlops-study-web-lmw", f"models/model_{save_model_name}_v1.pth")
            except NoCredentialsError:
                print("Failed S3 Successes")
