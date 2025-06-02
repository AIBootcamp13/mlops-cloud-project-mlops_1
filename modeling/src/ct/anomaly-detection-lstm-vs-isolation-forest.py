#!/usr/bin/env python
# coding: utf-8

# In[38]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from torch.utils.data import DataLoader, TensorDataset
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# # Context
# 
# In the Solar Energy Industry it is common to have **misproduction problems** regarding various topics such as dirty solar panels, inverter failures, sensor issues and more. In this Notebook I will compare two approaches. The first one using **Isolation Forest** and the second an **LSTM Autoencoder**, to see which approach is the most efficient to detect anomalies in an AC Power timeseries.

# In[2]:


generation1 = pd.read_csv('../../../data/train/Plant_1_Generation_Data.csv')
weather1 = pd.read_csv('../../../data/train/Plant_1_Weather_Sensor_Data.csv')
generation1['DATE_TIME'] = pd.to_datetime(generation1['DATE_TIME'], dayfirst=True)
weather1['DATE_TIME'] = pd.to_datetime(weather1['DATE_TIME'], dayfirst=False)


# In[3]:


generation1


# In[4]:


inverters = list(generation1['SOURCE_KEY'].unique())
print(f"total number of inverters {len(inverters)}")


# # Inverter level Anomally detection

# In[5]:


inverters[0]


# In[6]:


inv_1 = generation1[generation1['SOURCE_KEY']==inverters[0]]
mask = ((weather1['DATE_TIME'] >= min(inv_1["DATE_TIME"])) & (weather1['DATE_TIME'] <= max(inv_1["DATE_TIME"])))
weather_filtered = weather1.loc[mask]


# In[7]:


weather_filtered.shape


# In[10]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=inv_1["DATE_TIME"], y=inv_1["AC_POWER"],
                    mode='lines',
                    name='AC Power'))

fig.add_trace(go.Scatter(x=weather_filtered["DATE_TIME"], y=weather_filtered["IRRADIATION"],
                    mode='lines',
                    name='Irradiation', 
                    yaxis='y2'))

fig.update_layout(title_text="Irradiation vs AC POWER",
                  yaxis1=dict(title="AC Power in kW",
                              side='left'),
                  yaxis2=dict(title="Irradiation index",
                              side='right',
                              anchor="x",
                              overlaying="y"
                             ))

fig.write_image('../../../data/outputs/train/AC_power.png')


# ### Graph observations
# We can see that in June 7th and June 14th there are some misproduction areas that could be considered anomalies. Due to the fact that energy production should behave in a linear way to irradiation.

# In[11]:


df = inv_1.merge(weather_filtered, on="DATE_TIME", how='left')
df = df[['DATE_TIME', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
df


# ### Observations
# Here we can see how the Isolation Forest Model is behaving. The yellow dots show us the anomalies detected on the test dataset as well as the red squares that show us the anomalies detected on the training dataset. These points do not follow the contour pattern of the graph and we can clearly see that the yellow dots on the far left are the points from June 7th and June 14th.

# # LSTM Autoencoder approach

# In[12]:


df = df[["DATE_TIME", "AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
df_timestamp = df[["DATE_TIME"]]
df_ = df[["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]


# In[13]:


train_prp = .6
train = df_.loc[:df_.shape[0]*train_prp]
test = df_.loc[df_.shape[0]*train_prp:]


# In[34]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# In[35]:


X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42, shuffle=True)


# In[36]:


import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = 4  # same as L2 & L4 unit count

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

        # Repeat vector (same as RepeatVector in Keras)
        x = hidden.repeat(self.seq_len, 1, 1).permute(1, 0, 2)

        # Decode
        x, _ = self.decoder[0](x)
        x, _ = self.decoder[1](x)
        x = self.decoder[2](x)
        return x


# In[ ]:


epochs = 100
batch_size = 10
learning_rate = 1e-3

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor), batch_size=batch_size, shuffle=False)

model = LSTMAutoencoder(seq_len=X_train.shape[1], n_features=X_train.shape[2])
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_epoch_loss = 0
    for batch in train_loader:
        batch = batch[0]  # unpack from TensorDataset
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()
    
    avg_train_loss = train_epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch[0]
            output = model(batch)
            loss = criterion(output, batch)
            val_epoch_loss += loss.item()
    avg_val_loss = val_epoch_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), '../../../data/outputs/train/lstm.pth')


# In[40]:


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(len(train_losses))),
    y=train_losses,
    mode='lines',
    name='Train Loss'
))

fig.add_trace(go.Scatter(
    x=list(range(len(val_losses))),
    y=val_losses,
    mode='lines',
    name='Validation Loss'
))

fig.update_layout(
    title="Autoencoder MAE Loss over Epochs",
    xaxis=dict(title="Epoch"),
    yaxis=dict(title="MAE Loss"),
    width=800,
    height=500
)

fig.write_image('../../../data/outputs/train/Error_loss.png')


# In[41]:


import numpy as np
import pandas as pd

# 예측 (no grad & eval mode)
model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_pred = model(X_tensor)
    X_pred = X_pred.detach().numpy()

# reshape: (batch, seq_len, features) → (batch, features)
X_pred = X_pred[:, -1, :]  # 가장 마지막 시점의 예측값 사용

# inverse scaling
X_pred_inv = scaler.inverse_transform(X_pred)

# DataFrame 생성
X_pred_df = pd.DataFrame(X_pred_inv, columns=train.columns)


# In[42]:


scores = pd.DataFrame()
scores['AC_train'] = train['AC_POWER'].values[:len(X_pred_df)]
scores['AC_predicted'] = X_pred_df['AC_POWER']
scores['loss_mae'] = (scores['AC_train'] - scores['AC_predicted']).abs()


# In[43]:


fig = go.Figure(data=[go.Histogram(x=scores['loss_mae'])])
fig.update_layout(title="Error distribution", 
                 xaxis=dict(title="Error delta between predicted and real data [AC Power]"),
                 yaxis=dict(title="Data point counts"))
fig.write_image('../../../data/outputs/train/Error_distribution.png')


# In[44]:


model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    X_pred = model(X_tensor).detach().numpy()

X_pred = X_pred[:, -1, :]  # (batch, features)
X_pred_inv = scaler.inverse_transform(X_pred)
X_pred_df = pd.DataFrame(X_pred_inv, columns=train.columns)
X_pred_df.index = test.index


# In[45]:


# 예측 결과 DataFrame 복사
scores = X_pred_df.copy()

# datetime 컬럼 추가 (예: sliding window 적용 시 1893부터)
scores['datetime'] = df_timestamp.loc[scores.index]

# 실제값 추가
scores['real AC'] = test['AC_POWER'].values

# MAE 계산
scores['loss_mae'] = (scores['real AC'] - scores['AC_POWER']).abs()

# 이상치 기준 설정
scores['Threshold'] = 200
scores['Anomaly'] = np.where(scores['loss_mae'] > scores['Threshold'], 1, 0)


# In[46]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=scores['datetime'], 
                         y=scores['loss_mae'], 
                         name="Loss"))
fig.add_trace(go.Scatter(x=scores['datetime'], 
                         y=scores['Threshold'],
                         name="Threshold"))

fig.update_layout(title="Error Timeseries and Threshold", 
                 xaxis=dict(title="DateTime"),
                 yaxis=dict(title="Loss"))
fig.write_image('../../../data/outputs/train/Threshold.png')


# In[47]:


scores['Anomaly'].value_counts()


# In[48]:


anomalies = scores[scores['Anomaly'] == 1][['real AC']]
anomalies = anomalies.rename(columns={'real AC':'anomalies'})
scores = scores.merge(anomalies, left_index=True, right_index=True, how='left')


# In[53]:


scores[(scores['Anomaly']==1)&(scores['datetime'].notnull())].to_csv('../../../data/outputs/train/anomalies.csv')


# In[ ]:





# In[49]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=scores["datetime"], y=scores["real AC"],
                    mode='lines',
                    name='AC Power'))

fig.add_trace(go.Scatter(x=scores["datetime"], y=scores["anomalies"],
                    name='Anomaly', 
                    mode='markers',
                    marker=dict(color="red",
                                size=11,
                                line=dict(color="red",
                                          width=2))))

fig.update_layout(title_text="Anomalies Detected LSTM Autoencoder")

fig.write_image('../../../data/outputs/train/Anomaly.png')


# ## Conclusion
# 
# We see that the LSTM Autoencoder approach is a more efficient way to detect anomalies, againts the Isolation Forest approach, perhaps with a larger dataset the Isolation tree could outperform the Autoencoder, having a faster and pretty good model to detect anomalies. 
# 
# We can see from the Isolation Forest graph how the model is detecting anomalies, highlighting the datapoints from June 7th and June 14th.
# 

# In[ ]:




