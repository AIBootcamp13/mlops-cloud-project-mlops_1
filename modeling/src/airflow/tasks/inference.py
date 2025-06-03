def inference(project_path):
    import os
    import sys

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    temperature_df = pd.read_csv(os.path.join(project_path, 'data/TA_data.csv'))
    temperature_df_train = temperature_df[-365*5:-365].reset_index(drop=True)
    temperature_df_inference = temperature_df[-365:].reset_index(drop=True)
    temperature_df = temperature_df_inference

    temperature_columns = ['TA_AVG', 'TA_MAX', 'TA_MIN']


    temperature_df_timestamp = temperature_df[['날짜']].copy()
    temperature_df_timestamp.rename(columns={'날짜': 'datetime'}, inplace=True)
    temperature_df = temperature_df[temperature_columns]


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
    X = scaler.fit_transform(temperature_df.values)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    model_path = os.path.join(project_path, f"models/lstm_temperature_anomaly_detector.pth")
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
    X_pred_df = pd.DataFrame(X_pred_inv, columns=temperature_df.columns)
    X_pred_df.index = temperature_df.index

    for column in temperature_columns:
        scores = X_pred_df.copy()

        scores['datetime'] = pd.to_datetime(temperature_df_timestamp['datetime'], errors="coerce")
        scores['real'] = temperature_df[column].values
        scores['loss_mae'] = np.abs(scores['real'] - scores[column])
        scores['Threshold'] = 40
        scores['Anomaly'] = (scores['loss_mae'] > scores['Threshold']).astype(int)
        scores['anomalies'] = np.where(scores["Anomaly"] == 1, scores["real"], np.nan)

        scores = scores.sort_values("datetime").reset_index(drop=True)

        inference_output_dir = os.path.join(project_path, 'data/outputs/inference')
        os.makedirs(inference_output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(scores["datetime"], scores["loss_mae"], label="Loss")
        ax.plot(scores["datetime"], scores["Threshold"], label="Threshold", linestyle='--')

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        ax.set_title("Loss vs Threshold")
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(inference_output_dir, f"temperature_{column}_Threshold.png"))
        plt.close()

        cols = ['datetime'] + [col for col in scores.columns if col != 'datetime']
        scores = scores[cols]
        scores[scores["Anomaly"] == 1].to_csv(
            os.path.join(inference_output_dir, f'temperature_{column}_anomalies.csv'),
            index=False
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(scores["datetime"], scores["real"], label=column)
        if scores["Anomaly"].sum() > 0:
            mask = scores["Anomaly"] == 1
            ax.scatter(scores.loc[mask, "datetime"], scores.loc[mask, "anomalies"],
                    color="red", label="Anomaly", s=25)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        ax.set_title("Anomalies Detected (Inference)")
        ax.set_xlabel("Datetime")
        ax.set_ylabel(column)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(inference_output_dir, f"temperature_{column}_Anomaly.png"))
        plt.close()