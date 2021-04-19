import pandas as pd
import numpy as np
from transform_df import *
from bucket import *

from pandas_datareader import data,wb
from technical_indicators_lib import OBV , CCI , CHV , CMF , DPO, EMA , EMV , MACD ,MFI ,MI, PVT, RSI, EMV
import json, boto3
import matplotlib.dates as dates
from datetime import date , timedelta
import streamlit as st
pd.options.mode.chained_assignment = None 

start = pd.to_datetime("2011-06-01")
end = pd.to_datetime(date.today())

GOLD = data.DataReader("GOLD", 'yahoo',start,end)
LUV = data.DataReader("LUV", 'yahoo',start,end)
BBVA = data.DataReader("BBVA", 'yahoo',start,end)
TEL = data.DataReader("TEL", 'yahoo',start,end)
TAK = data.DataReader("TAK", 'yahoo',start,end)
EPD = data.DataReader("EPD", 'yahoo',start,end)
GD = data.DataReader("GD", 'yahoo',start,end)
AAPL = data.DataReader("AAPL", 'yahoo',start,end)
TSLA = data.DataReader("TSLA", 'yahoo',start,end)
AEP = data.DataReader("AEP", 'yahoo',start,end)

ticker_list = [GOLD,LUV,BBVA,TEL,TAK, EPD,GD,AAPL,TSLA,AEP]

liss = [OBV() , CCI() , CHV(), CMF() , DPO(), EMA() , EMV() , MACD() ,MI(), PVT(), RSI()]

close = []
target = []
ohlcv = []
for i in ticker_list:
    data = get_df(i,liss)
    print(data.shape)
    target.append(data["dfair"].values)
    ohlcv.append(data[["open","high","low","adjclose","volume"]])
    close.append(data.drop(["dfair","open","high","low","close","adjclose","fair","volume"],axis=1))
    #print(data.shape)
#target = np.stack(target, axis =1)
#df = np.stack(close, axis =2)
ohlcv = np.stack(ohlcv,axis=1)
df = np.stack(close, axis=1)
df.shape
#s3 = boto3.resource("s3", region_name='us-east-2',
    #aws_access_key_id="AKIAVVZS666ZXRWNM6HM",
    #aws_secret_access_key="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").Bucket("data-close")
#json.load_s3 = lambda f: json.load(s3.Object(key=f).get()["Body"])

json.dump_s3 = lambda obj, f: S3().Object(key=f).put(Body=json.dumps(obj, sort_keys=True, default=str).encode('UTF-8'))
json.dump_s3_t = lambda obj, f: S3_t().Object(key=f).put(Body=json.dumps(obj, sort_keys=True, default=str).encode('UTF-8'))


df_2D = pd.DataFrame(np.stack(close, axis = 1).reshape(len(df), -1))
print(df_2D.shape)

#df_dict = df_2D.to_dict('records')
df_2D.to_csv("close.csv")

#json.dump_s3(df_dict, "key")

target_df = pd.DataFrame(target)
target_df.to_csv("target.csv")
#json.dump_s3_t(target_df, "key")
ohlcv_2D = pd.DataFrame(np.stack(ohlcv, axis = 1).reshape(len(df),-1))
ohlcv_2D.to_csv("ohlcv.csv")

print("data collected")