import talib
import pandas as pd
import json
import scipy.stats
import math
import os
import re
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, pearsonr, spearmanr, ttest_1samp, ttest_ind
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

from sklearn.linear_model import Ridge, RidgeCV, LassoCV, LogisticRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.ensemble import BaggingRegressor

import random
from sklearn import model_selection

from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from lib.utils.create_imbalance_bar import Tick_Imbalance

# ライブラリのインポート
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM,Conv2D,MaxPooling2D

from keras.optimizers import Adam
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import tensorflow
from keras.layers import Flatten
from keras.utils import np_utils

from tensorflow.python.keras.models import load_model
from scipy import optimize
import pylab as pl
from fractions import Fraction

import numpy as np
class NN:
    def __init__(self,input_shape,save_path,max_length):
        self.save_path = save_path
        self.input_shape = input_shape
        self.max_length = max_length

        #CNN
        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=(self.max_length,)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(512))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(512))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3))
        self.model.add(Activation('softmax'))

        opt = tensorflow.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
      
    def model_train(self,x, y):
        self.model.fit(x, y, batch_size=32, epochs=500)

        self.model.save(self.save_path)

        return self.model


    def model_eval(self, x, y):
        scores = self.model.evaluate(x, y, verbose=1)
        print('Test Loss: ', scores[0])
        print('Test Accuracy: ', scores[1])

# 例としてtalibで特徴量をいくつか生成

def calc_features(df):
    open = df['op']
    high = df['hi']
    low = df['lo']
    close = df['cl']
    
    orig_columns = df.columns

    print('calc talib overlap')
    hilo = (df['hi'] + df['lo']) / 2
    df['BBANDS_upperband'], df['BBANDS_middleband'], df['BBANDS_lowerband'] = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['BBANDS_upperband'] -= hilo
    df['BBANDS_middleband'] -= hilo
    df['BBANDS_lowerband'] -= hilo
    df['DEMA'] = talib.DEMA(close, timeperiod=30) - hilo
    df['EMA'] = talib.EMA(close, timeperiod=30) - hilo
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close) - hilo
    df['KAMA'] = talib.KAMA(close, timeperiod=30) - hilo
    df['MA'] = talib.MA(close, timeperiod=30, matype=0) - hilo
    df['MIDPOINT'] = talib.MIDPOINT(close, timeperiod=14) - hilo
    df['SMA'] = talib.SMA(close, timeperiod=30) - hilo
    df['T3'] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
    df['TEMA'] = talib.TEMA(close, timeperiod=30) - hilo
    df['TRIMA'] = talib.TRIMA(close, timeperiod=30) - hilo
    df['WMA'] = talib.WMA(close, timeperiod=30) - hilo

    print('calc talib momentum')
    df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
    df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    df['AROON_aroondown'], df['AROON_aroonup'] = talib.AROON(high, low, timeperiod=14)
    df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
    df['BOP'] = talib.BOP(open, high, low, close)
    df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    df['DX'] = talib.DX(high, low, close, timeperiod=14)
    df['MACD_macd'], df['MACD_macdsignal'], df['MACD_macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    # skip MACDEXT MACDFIX たぶん同じなので
    df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
    df['MOM'] = talib.MOM(close, timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
    df['RSI'] = talib.RSI(close, timeperiod=14)
    df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['TRIX'] = talib.TRIX(close, timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)


    print('calc talib vola')
    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    df['NATR'] = talib.NATR(high, low, close, timeperiod=14)
    df['TRANGE'] = talib.TRANGE(high, low, close)

    print('calc talib cycle')
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
    df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(close)
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

    print('calc talib stats')
    df['BETA'] = talib.BETA(high, low, timeperiod=5)
    df['CORREL'] = talib.CORREL(high, low, timeperiod=30)
    df['LINEARREG'] = talib.LINEARREG(close, timeperiod=14) - close
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
    df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
    df['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1)

    return df

features = sorted([
    'ADX',
    'ADXR',
    'APO',
   'DX',
    'MACD_macd',
    'MACD_macdsignal',
   'RSI',
    'STOCH_slowk',
    'STOCH_slowd',
    'STOCHF_fastk',
    'STOCHRSI_fastd',
    'ULTOSC',
    'WILLR',
#     'NATR',
    'HT_DCPERIOD',
   'BETA',
   'STDDEV',
   'DEMA',
    'KAMA',
])


data = pd.read_csv('tickdata_gmo_xrp.csv')
tick_imbalance = Tick_Imbalance(0.0001,0.0001,1,9000)
a = tick_imbalance.get_b(data['price'])
b = tick_imbalance.get_bar_ids(a*np.array(data['amount']*data['price'])[1:])
data = data.iloc[1:].reset_index(drop=True)
data['tick'] = b

DataFrame = data[['tick','price']].groupby(['tick']).min().rename(columns={'price':'lo'})
DataFrame['hi'] = data[['tick','price']].groupby(['tick']).max()
DataFrame['op'] = data[['tick','price']].groupby(['tick']).first()
DataFrame['cl'] = data[['tick','price']].groupby(['tick']).last()

df = calc_features(DataFrame)
df.to_pickle('df_features.pkl')

df = pd.read_pickle('df_features.pkl')
df = df.dropna().reset_index(drop=True)

max_length = 300
width = 0.02
y_data = []
for i in range(len(df)-max_length):
    sep_price = df.loc[i:i+max_length].reset_index(drop=True)
    upper_price = sep_price[sep_price['hi'] > sep_price['op'].iloc[0] * (1+width)]
    if len(upper_price) > 0:
        upper_time = upper_price.index[0]
    else:
        upper_time = max_length + 1
    lower_price = sep_price[sep_price['lo'] < sep_price['op'].iloc[0] * (1-width)]
    if len(lower_price) > 0:
        lower_time = lower_price.index[0]
    else:
        lower_time = max_length + 1
    
    if  lower_time > upper_time:
        output = 2#上昇
    else:
        output = 1#下落
    
    if max_length < upper_time and max_length < lower_time:
        output = 0#変動なし
    
    y_data.append(output)


u, counts = np.unique(np.array(y_data), return_counts=True)
print(counts)

y_categorical_data = []
for i in y_data:
    if i == 0:
        data = [0,0,1]
    elif i == 1:
        data = [0,1,0]
    else:
        data = [0,0,1]
    y_categorical_data.append(data)

x_df = df.copy()
x_df = df[features].copy()

x_data = []
input_len = 100
for i in range(len(df)-input_len):
    indata = scipy.stats.zscore(x_df.loc[i:i+input_len])
    x_data.append(np.array(indata).reshape(-1).tolist())


y_data_ = np.array(y_categorical_data)

y_data_ = y_data_[input_len:]
x_data = x_data[:-max_length]


x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data,y_data_)

nn = load_model('imbalance_triple_barrier.h5')
#nn.model_train(np.array(x_train),np.array(y_train))
scores =nn.evaluate(np.array(x_data)[-100:], np.array(y_data_)[-100:], verbose=1)
print('Test Loss: ', scores[0])
print('Test Accuracy: ', scores[1])


