import numpy as np
import json
import boto3

def get_df(data,liss):
    data.columns = ['high', 'low', 'open', 'close', 'volume', 'adjclose']
    data["ma200"] = data["open"].rolling(200).mean()

    for i in liss:
        df = i.get_value_df(data)

    df["fair"] = df["open"].ewm(span=5).mean().shift(1)
    df["dfair"] = np.log(df.open) - np.log(df.open.shift(1))
    #df["dfair"] = df["fair"].pct_change()
    df = df.dropna()

    print(df.shape)
    return df

