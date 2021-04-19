import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import streamlit as st
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
pd.options.mode.chained_assignment = None 
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv3D, Flatten, Dropout, concatenate,BatchNormalization , LSTM , GRU, RNN, Conv2D, Conv1D, GlobalMaxPooling1D, TimeDistributed,Input,Bidirectional, ConvLSTM2D,GlobalAveragePooling1D,Attention
from keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
#from stqdm import stqdm

from chuncks import * 
from models import *
from bucket import *
from transform_df import *

print(gold["open"])

GOLD = data.DataReader("GOLD", 'yahoo',start,end)

print(gold["open"])
