import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def train():
    gen = pd.read_csv("../../../data/train/Plant_1_Generation_Data.csv")
    weather = pd.read_csv("../../../data/train/Plant_1_Weather_Sensor_Data.csv")

    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], dayfirst=True)
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])

    inverter = gen[gen["SOURCE_KEY"] == gen["SOURCE_KEY"].unique()[0]]
    weather_filtered = weather[
        (weather["DATE_TIME"] >= inverter["DATE_TIME"].min()) &
        (weather["DATE_TIME"] <= inverter["DATE_TIME"].max())
    ]

    df = inverter.merge(weather_filtered, on="DATE_TIME", how="left")
    df = df[["DATE_TIME", "AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    df_timestamp = df["DATE_TIME"]
    df_ = df.drop(columns=["DATE_TIME"])

    train_size = int(len(df_) * 0.6)
    train_df, test_df = df_[:train_size], df_[train_size:]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df)
    X_test = scaler.transform(test_df)

    X_train = X_train.reshape(-1, 1, X_train.shape[1])
    X_test = X_test.reshape(-1, 1, X_test.shape[1])

    X_train, X_val = train_test_split(X_train, test_size=0.2, shuffle=True, random_state=42)

    class LSTMAutoencoder(nn.Module):
        def __init__(self, seq_len, n_features):
            super(LSTMAutoencoder, self).__init__()
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
            x, _ = self.encoder[0](x)
            x, (hidden, _) = self.encoder[1](x)
            x = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)
            x, _ = self.decoder[0](x)
            x, _ = self.decoder[1](x)
            x = self.decoder[2](x)
            return x

    model = LSTMAutoencoder(seq_len=1, n_features=X_train.shape[2])
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100
    batch_size = 10

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32)), batch_size=batch_size)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                output = model(batch)
                val_loss += criterion(output, batch).item()
        val_losses.append(val_loss / len(val_loader))

        print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    os.makedirs("../../../data/outputs/train", exist_ok=True)
    torch.save(model.state_dict(), "../../../data/outputs/train/lstm.pth")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Autoencoder MAE Loss over Epochs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../../data/outputs/train/Error_loss.png")
    plt.close()

    with torch.no_grad():
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        pred = model(X_tensor).numpy()
    pred = pred[:, -1, :]
    X_pred_inv = scaler.inverse_transform(pred)
    pred_df = pd.DataFrame(X_pred_inv, columns=train_df.columns)

    scores = pd.DataFrame()
    scores["AC_train"] = train_df["AC_POWER"].values[:len(pred_df)]
    scores["AC_predicted"] = pred_df["AC_POWER"]
    scores["loss_mae"] = (scores["AC_train"] - scores["AC_predicted"]).abs()

    plt.figure(figsize=(8, 4))
    plt.hist(scores["loss_mae"], bins=50)
    plt.title("Error Distribution (Train)")
    plt.xlabel("MAE (AC_POWER)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("../../../data/outputs/train/Error_distribution.png")
    plt.close()

    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        X_pred = model(X_tensor).numpy()

    X_pred = X_pred[:, -1, :]
    X_pred_inv = scaler.inverse_transform(X_pred)
    X_pred_df = pd.DataFrame(X_pred_inv, columns=train_df.columns)
    X_pred_df.index = test_df.index

    scores = X_pred_df.copy()
    scores["datetime"] = df_timestamp.iloc[test_df.index]
    scores["real AC"] = test_df["AC_POWER"].values
    scores["loss_mae"] = (scores["real AC"] - scores["AC_POWER"]).abs()
    scores["Threshold"] = 200
    scores["Anomaly"] = (scores["loss_mae"] > scores["Threshold"]).astype(int)

    plt.figure(figsize=(12, 5))
    plt.plot(scores["datetime"], scores["loss_mae"], label="Loss")
    plt.plot(scores["datetime"], scores["Threshold"], label="Threshold", linestyle="--")
    plt.title("Loss vs Threshold")
    plt.xlabel("Datetime")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../data/outputs/train/Threshold.png")
    plt.close()

    anomalies = scores[scores["Anomaly"] == 1][["real AC"]].rename(columns={"real AC": "anomalies"})
    scores = scores.merge(anomalies, left_index=True, right_index=True, how="left")
    scores[scores["Anomaly"] == 1].to_csv("../../../data/outputs/train/anomalies.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(scores["datetime"], scores["real AC"], label="AC Power")
    plt.scatter(scores["datetime"], scores["anomalies"], color="red", label="Anomaly", s=25)
    plt.title("Anomalies Detected (Test)")
    plt.xlabel("Datetime")
    plt.ylabel("AC Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../data/outputs/train/Anomaly.png")
    plt.close()

if __name__ == "__main__":
    train()