import sys
import pandas as pd
import numpy as np
import scipy
import talib
from scipy.stats import entropy

sys.path.append("../../")
from sklearn import model_selection
from logics.entropy import Entropy_alg, Calc_entropy
from modules.loads import Load_data, Reshape_data, Probably_distoribution, Save_image, Tick_Imbalance, Discord_system
from modules.backtest import BackTest
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))


from tensorflow.keras import optimizers
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM,Conv2D,MaxPooling2D, Dropout

import tensorflow
from keras.layers import Flatten
from keras.utils import np_utils

from tensorflow.python.keras.models import load_model
from scipy import optimize

class NN:
    def __init__(self,save_path,max_length):
        self.save_path = save_path
        self.max_length = max_length

        #CNN
        self.model = Sequential()
        self.model.add(LSTM(512,input_shape=(1,3),return_sequences=True))
        self.model.add(Activation('relu'))
        self.model.add(LSTM(512,return_sequences=True))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))

        self.model.add(LSTM(512,return_sequences=True))
        self.model.add(Activation('relu'))
        self.model.add(LSTM(512,return_sequences=True))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))

        self.model.add(LSTM(512,return_sequences=False))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512))

        self.model.add(Activation('softmax'))
        self.model.add(Dense(3))

        opt = optimizers.RMSprop(lr=1e-4)

        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
      
    def model_train(self,x, y):
        self.model.fit(x, y, batch_size=32, epochs=20)

        self.model.save(self.save_path)

        return self.model


    def model_eval(self, x, y):
        scores = self.model.evaluate(x, y, verbose=1)
        print('Test Loss: ', scores[0])
        print('Test Accuracy: ', scores[1])



class Environments:
    def __init__(self, symbol, read_num):
        self.symbol = symbol
        self.read_num = read_num
        self.input_len = 1

        self.nn = NN(root_path/ 'data' /'entropy'/  (self.symbol + 'gamma_lstm.h5'), 12* self.input_len)

    def calc_features(self, df):
        open = df['op']
        high = df['hi']
        low = df['lo']
        close = df['cl']
        hilo = (df['hi'] + df['lo']) / 2
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1)
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['MOM'] = talib.MOM(close, timeperiod=10)
        df['WMA'] = talib.WMA(close, timeperiod=30) - hilo
        df['AROON_aroondown'], df['AROON_aroonup'] = talib.AROON(high, low, timeperiod=14)
        df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)

        
        return df

        
    def run(self):
        DataFrame = pd.read_pickle('gammas_and_rewards.pkl')
        #DataFrame = self.calc_features(DataFrame)
        DataFrame = DataFrame.dropna()
        DataFrame = DataFrame[DataFrame['buyside_gamma'] < -0.1]
        DataFrame = DataFrame[DataFrame['sellside_gamma'] < -0.1]

        print(DataFrame)
        for i in range(len(DataFrame) - self.input_len):
            data = DataFrame[['sigma', 'buyside_alpha', 'sellside_alpha']].iloc[i:i + self.input_len]
            #data = scipy.stats.zscore(data.values)
            indata = np.array([data.values])#.reshape(-1)
            y = np.array([DataFrame[['buyside_gamma', 'sellside_gamma', 'reward']].iloc[i+self.input_len]])
            if i == 0:
                x_data = indata
                y_data = y
            else:
                x_data = np.append(x_data,indata,axis=0)
                y_data = np.append(y_data, y, axis = 0)
                #x_data = np.vstack([x_data,indata])
            #print(x_data.shape)
            if i % 10000 == 0:
                print(i)

        np.save('x.npy', x_data)
        np.save('y.npy', y_data)

    def get_y(self):
        DataFrame = pd.read_pickle(root_path/ 'data' / (self.symbol + '_entropy.pkl')).rename(columns={'sum_entorpy':'sum_entropy'})
        DataFrame['price'] = DataFrame['price'].apply(np.log)
        DataFrame = DataFrame[['price', 'amount', 'diff_entropy', 'sum_entropy']]
        DataFrame = self.get_imbalance_bar(DataFrame)
        y_data = self.triple_barrier(DataFrame)
        y_data = y_data[self.input_len:]

        np.save(root_path/ 'data' /'entropy'/  (self.symbol + '_entropy_y.npy'), y_data)

    def learn(self):
        x_data = np.load('x.npy')
        y_data = np.load('y.npy')
        print(x_data)
        print(y_data)

        x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data,y_data)

        self.nn.model_train(np.array(x_train),np.array(y_train))
        self.nn.model_eval(np.array(x_test),np.array(y_test))

if __name__ == "__main__":

    env = Environments('gmo_xrp', 2000000)
    #env.get_pkl()
    #env.learn()
    env.learn()
