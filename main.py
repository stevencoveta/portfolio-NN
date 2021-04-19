import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import streamlit as st
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
#pd.options.mode.chained_assignment = None 
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv3D, Flatten, Dropout, concatenate,BatchNormalization , LSTM , GRU, RNN, Conv2D, Conv1D, GlobalMaxPooling1D, TimeDistributed,Input,Bidirectional, ConvLSTM2D,GlobalAveragePooling1D,Attention
from keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from stqdm import stqdm

from chuncks import * 
from models import *
from bucket import *
from transform_df import *

#json.load_s3 = lambda f: json.load(S3().Object(key=f).get()["Body"])
#json.load_s3_t = lambda f: json.load(S3_t().Object(key=f).get()["Body"])

print("loading buckets")

#loaded = json.load_s3('key')
#target = pd.DataFrame(json.load_s3_t('key'))
target = pd.read_csv("target.csv",index_col=0)
#loaded.to_csv("l.csv")

#target = pd.read_csv("target.csv")
loaded_arr = pd.read_csv("close.csv", index_col=0)
arr = np.stack(loaded_arr)
close = loaded_arr.values.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // 31, 31)
  
print(close.shape) 

loaded_arr = pd.read_csv("ohlcv.csv",index_col=0)
arr = np.stack(loaded_arr)

ohlcv =loaded_arr.values.reshape(
    loaded_arr.shape[0], 5,loaded_arr.shape[1] // 5)
print(ohlcv.shape)



pred = []
num=0.7
for i in stqdm(range(10),desc="This is a slow task",):
    scale = StandardScaler()
    scaler = StandardScaler()
    scaler_x = StandardScaler()
    scaler_x1 = StandardScaler()

    dff = pd.DataFrame(close[:,i])
    targett = (pd.DataFrame(target).T[i].values.reshape(-1,1))
  
    X_train = scaler_x.fit_transform(dff[:int(len(dff) * num)])
    X_test = scaler_x1.fit_transform(dff[int(len(dff) * num):])
    y_train = scaler.fit_transform(targett[:int(len(dff) * num)].reshape(-1,1))
    y_test = scale.fit_transform(targett[int(len(dff) * num):].reshape(-1,1))
    
    #X_train = scaler_col(X_train)
    #X_test = scaler_col(X_test)
    
    X_train = autoencoder(X_train)
    X_test = autoencoder(X_test)
    
    print(X_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(y_train.shape)

    
    X_train, y_train = train_chunck(X_train, y_train)
    X_test, y_test = test_chunck(X_test, y_test)
    print(X_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(y_train.shape)
    #models.load_weights("models/model.h5")
    #y_pred = models.predict(X_test)
    #print(y_pred)
    y_pred = model_bilstm_get_prediction(X_train,y_train,X_test,y_test)
    #print(y_pred)
    #y_pred = lstm(X_train,y_train,X_test,y_test)
    #pred.append(y_pred)
    #print(pred)
    pred.append(scale.inverse_transform(y_pred))

portfolio_pred = pd.DataFrame(np.stack(pred,axis=1).reshape(len(pred[0]),ohlcv.shape[2]))
print(portfolio_pred)
#st.progress
#st.line_chart(portfolio_pred)

closes = []

closes_dif= []
for i in range(ohlcv.shape[2]):
    closes.append(pd.DataFrame(ohlcv[:,3:4,i:i+1].reshape(ohlcv.shape[0],1)))

closes = pd.concat(closes, axis =1)
closes.columns = list(range(ohlcv.shape[2]))
closes = closes[-len(y_test):].pct_change()
closes = closes.reset_index(drop=True)
closes = closes.fillna(0)


kmeans = KMeans(n_clusters=4)
clusters = []
for i in range(len(closes)):
    kmeans.fit(closes.iloc[i].values.reshape(-1,1))
    clusters.append(kmeans.labels_)
best_cluster = []



for i in range(len(clusters)):
    mlo = pd.DataFrame(portfolio_pred.iloc[i], index = clusters[i])
    best_cluster.append(mlo.groupby(mlo.index).sum().idxmax().values[0])
clusters = pd.DataFrame(clusters)



port = []
for i in range(len(clusters)):
    price = []
    for j in range(len(clusters.columns)):
        if clusters.iloc[i][j] == best_cluster[i]:
            price.append(closes.iloc[i][j])
        else:
            price.append(0)
    port.append(price)
closes_filtered = pd.DataFrame(port)


wei = []
for i in range(len(clusters)):
    price = []
    for j in range(len(clusters.columns)):
        if clusters.iloc[i][j] == best_cluster[i]:
            price.append(portfolio_pred.iloc[i][j])
        else:
            price.append(0)
    wei.append(price)
pred_filtered = pd.DataFrame(wei)


allocation = []
for (i,row1) , (row2) in zip(abs(pred_filtered).iterrows(),abs(pred_filtered).sum(axis=1)):
    allocation.append(row1 / row2)
allocation = pd.DataFrame(allocation)

portfolio_allocator = closes_filtered * allocation 

names_tickers = ["GOLD","LUV","BBVA","TEL","TAK", "EPD","GD","AAPL","TSLA","AEP"]

#plt.plot(portfolio_allocator.cumsum())
#plt.show
st.line_chart(portfolio_allocator.cumsum())