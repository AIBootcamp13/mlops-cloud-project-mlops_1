def inference():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    import torch
    import torch.nn as nn
    import os
    import sys
    import pytz
    from datetime import datetime

    data_root_path = "dags/data"

    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst).strftime('%Y%m%d_%H%M%S')

    generation1 = pd.read_csv(os.path.join(data_root_path, 'inference', 'Plant_1_Generation_Data.csv'))
    weather1 = pd.read_csv(os.path.join(data_root_path, 'inference', 'Plant_1_Weather_Sensor_Data.csv'))

    generation1['DATE_TIME'] = pd.to_datetime(generation1['DATE_TIME'], dayfirst=True)
    weather1['DATE_TIME'] = pd.to_datetime(weather1['DATE_TIME'], dayfirst=False)

    inverters = list(generation1['SOURCE_KEY'].unique())
    inv_1 = generation1[generation1['SOURCE_KEY'] == inverters[0]]
    mask = (weather1['DATE_TIME'] >= min(inv_1["DATE_TIME"])) & (weather1['DATE_TIME'] <= max(inv_1["DATE_TIME"]))
    weather_filtered = weather1.loc[mask]

    df = inv_1.merge(weather_filtered, on="DATE_TIME", how='left')
    df = df[["DATE_TIME", "AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    df_timestamp = df[["DATE_TIME"]]
    df_ = df.drop(columns=["DATE_TIME"])

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

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_.values)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model_path = os.path.join(data_root_path, 'outputs', 'train', f'lstm.pth')
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model = LSTMAutoencoder(seq_len=1, n_features=X.shape[2])
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        X_pred = model(X_tensor).detach().numpy()

    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred_inv = scaler.inverse_transform(X_pred)
    X_pred_df = pd.DataFrame(X_pred_inv, columns=df_.columns)
    X_pred_df.index = df_.index

    scores = X_pred_df.copy()
    scores['datetime'] = df_timestamp.values
    scores['real AC'] = df_['AC_POWER'].values
    scores['loss_mae'] = np.abs(scores['real AC'] - scores['AC_POWER'])
    scores['Threshold'] = 200
    scores['Anomaly'] = (scores['loss_mae'] > scores['Threshold']).astype(int)

    anomalies = scores[scores['Anomaly'] == 1][['real AC']].rename(columns={'real AC': 'anomalies'})
    scores = scores.merge(anomalies, left_index=True, right_index=True, how='left')

    inference_output_dir = os.path.join(data_root_path, 'outputs', 'inference')
    os.makedirs(inference_output_dir, exist_ok=True)
    scores[scores['Anomaly'] == 1].to_csv(os.path.join(inference_output_dir, f'{now}_anomalies.csv'), index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(scores['datetime'], scores['real AC'], label='AC Power', color='blue')
    plt.scatter(scores['datetime'], scores['anomalies'], color='red', label='Anomaly', s=30)
    plt.title("Anomalies Detected by LSTM Autoencoder")
    plt.xlabel("Datetime")
    plt.ylabel("AC Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(inference_output_dir, f'{now}_Anomaly.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(inv_1["DATE_TIME"], inv_1["AC_POWER"], label="AC Power", color="green")
    plt.plot(weather_filtered["DATE_TIME"], weather_filtered["IRRADIATION"], label="Irradiation", color="orange")
    plt.title("Irradiation vs AC POWER")
    plt.xlabel("Datetime")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(inference_output_dir, f'{now}_AC_power.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    inference()