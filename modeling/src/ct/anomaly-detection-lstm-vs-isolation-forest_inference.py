#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# # Context
# 
# In the Solar Energy Industry it is common to have **misproduction problems** regarding various topics such as dirty solar panels, inverter failures, sensor issues and more. In this Notebook I will compare two approaches. The first one using **Isolation Forest** and the second an **LSTM Autoencoder**, to see which approach is the most efficient to detect anomalies in an AC Power timeseries.

# In[2]:


generation1 = pd.read_csv('../../../data/inference/Plant_1_Generation_Data.csv')
weather1 = pd.read_csv('../../../data/inference/Plant_1_Weather_Sensor_Data.csv')
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


# In[8]:


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

fig.write_image('../../../data/outputs/inference/AC_power.png')


# ### Graph observations
# We can see that in June 7th and June 14th there are some misproduction areas that could be considered anomalies. Due to the fact that energy production should behave in a linear way to irradiation.

# In[9]:


df = inv_1.merge(weather_filtered, on="DATE_TIME", how='left')
df = df[['DATE_TIME', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
df


# ### Observations
# Here we can see how the Isolation Forest Model is behaving. The yellow dots show us the anomalies detected on the test dataset as well as the red squares that show us the anomalies detected on the training dataset. These points do not follow the contour pattern of the graph and we can clearly see that the yellow dots on the far left are the points from June 7th and June 14th.

# # LSTM Autoencoder approach

# In[10]:


df = df[["DATE_TIME", "AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
df_timestamp = df[["DATE_TIME"]]
df_ = df[["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]


# In[11]:


# train_prp = .6
# train = df_.loc[:df_.shape[0]*train_prp]
# test = df_.loc[df_.shape[0]*train_prp:]


# In[12]:


# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(train)
# X_test = scaler.transform(test)
# X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
# X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
# print(f"X_train shape: {X_train.shape}")
# print(f"X_test shape: {X_test.shape}")


# In[13]:


# X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42, shuffle=True)


# In[14]:


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


# In[15]:


scaler = MinMaxScaler()
X = scaler.fit_transform(df_.values)
X = X.reshape(X.shape[0], 1, X.shape[1])  # (samples, seq_len=1, features)

# 텐서 변환
X_tensor = torch.tensor(X, dtype=torch.float32)

model = LSTMAutoencoder(seq_len=1, n_features=X.shape[2])
model.load_state_dict(torch.load('../../../data/outputs/train/lstm.pth', weights_only=True))
model.eval()

with torch.no_grad():
    X_pred = model(X_tensor).detach().numpy()

X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred_inv = scaler.inverse_transform(X_pred)
X_pred_df = pd.DataFrame(X_pred_inv, columns=df_.columns)
X_pred_df.index = df_.index

scores = X_pred_df.copy()
scores['datetime'] = df_timestamp
scores['real AC'] = df_['AC_POWER'].values
scores['loss_mae'] = (scores['real AC'] - scores['AC_POWER']).abs()
scores['Threshold'] = 200
scores['Anomaly'] = np.where(scores['loss_mae'] > scores['Threshold'], 1, 0)


# In[47]:


scores['Anomaly'].value_counts()


# In[48]:


anomalies = scores[scores['Anomaly'] == 1][['real AC']]
anomalies = anomalies.rename(columns={'real AC':'anomalies'})
scores = scores.merge(anomalies, left_index=True, right_index=True, how='left')


# In[ ]:


scores[(scores['Anomaly']==1)&(scores['datetime'].notnull())].to_csv('../../../data/outputs/inference/anomalies.csv')


# In[ ]:





# In[ ]:


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

fig.write_image('../../../data/outputs/inference/Anomaly.png')


# ## Conclusion
# 
# We see that the LSTM Autoencoder approach is a more efficient way to detect anomalies, againts the Isolation Forest approach, perhaps with a larger dataset the Isolation tree could outperform the Autoencoder, having a faster and pretty good model to detect anomalies. 
# 
# We can see from the Isolation Forest graph how the model is detecting anomalies, highlighting the datapoints from June 7th and June 14th.
# 

# In[ ]:




