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
ROOT_PATH = Path(os.environ['ROOT_PATH'])

class Environments:
    def __init__(self):
        self.symbol = 'gmo'
        self.read_num = 2500000
        self.ent_span = 50000
        self.load_data = Load_data()
        self.reshape_data = Reshape_data()
        self.discord_system = Discord_system()
        self.probably_distoribution = Probably_distoribution()
        self.probably_distoribution.is_CDF = False
        self.save_image = Save_image(self.symbol, self.ent_span)

    def run3(self):
        DataFrame = self.load_data.read_from_sql(('tickdata_' + self.symbol), self.read_num)
        #DataFrame['amount'] = DataFrame['amount'] * 100
        #DataFrame = self.reshape_data.groupby_switch(DataFrame)
        buyside_p_data = []
        sellside_p_data = []
        buyside_data = DataFrame[DataFrame['direction'] == 'BUY']['amount']

        x_b,y_b = self.probably_distoribution.from_DataFrame_to_trainnp(buyside_data)
        plt.plot(x_b,y_b)
        plt.xlim(0,0.15)
        self.png_path = ROOT_PATH / 'data' / 'entropy' / 'raw.png'
        plt.savefig(self.png_path)
        self.discord_system.push_image(self.png_path)


    def run(self):
        DataFrame = self.load_data.read_from_sql(('tickdata_' + self.symbol), self.read_num)
        #DataFrame['amount'] = DataFrame['amount'] * 100
        #DataFrame = self.reshape_data.groupby_switch(DataFrame)
        buyside_p_data = []
        sellside_p_data = []
        for i in range(len(DataFrame)- self.ent_span):
            if i % int(self.ent_span/100) == 0:
                now_datas = DataFrame.iloc[:i+self.ent_span]
                buyside_data = now_datas[now_datas['direction'] == 'BUY']['amount'].iloc[-self.ent_span:]
                sellside_data = now_datas[now_datas['direction'] == 'SELL']['amount'].iloc[-self.ent_span:]

                x_b,y_b = self.probably_distoribution.from_DataFrame_to_trainnp(buyside_data)
                try:
                    buyside_params = self.probably_distoribution.curve_fitting(x_b,y_b)
                    buyside_continuous_ent = self.probably_distoribution.continuous_entropy(*buyside_params)
                except:
                    buyside_continuous_ent = 0

                x_b,y_b = self.probably_distoribution.from_DataFrame_to_trainnp(sellside_data)
                try:
                    sellside_params = self.probably_distoribution.curve_fitting(x_b,y_b)
                    sellside_continuous_ent = self.probably_distoribution.continuous_entropy(*sellside_params)
                except:
                    sellside_continuous_ent = 0

            if i % int(10000) == 0:
                print(i)

            buyside_p_data.append(buyside_continuous_ent)
            sellside_p_data.append(sellside_continuous_ent)

        DataFrame = DataFrame.iloc[:-self.ent_span]
        DataFrame['buyside_con_ent'] = buyside_p_data
        DataFrame['sellside_con_ent'] = sellside_p_data

        pd.to_pickle(DataFrame,ROOT_PATH / 'data' / (self.symbol + '_entropy.pkl'))
        dic = {'price': DataFrame['price'],
               'buyside_ent': DataFrame['buyside_con_ent'],
               'sellside_ent': DataFrame['sellside_con_ent'],
               'diff_ent': DataFrame['buyside_con_ent'] - DataFrame['sellside_con_ent'],
               'sum_ent': DataFrame['buyside_con_ent'] + DataFrame['sellside_con_ent']}

        self.save_image.multi_images(dic)


    def run2(self):
        DataFrame = pd.read_pickle(ROOT_PATH / 'data' / (self.symbol + '_entropy.pkl'))
        dic = {'price': DataFrame['price'],
               'buyside_ent': DataFrame['buyside_con_ent'],
               'sellside_ent': DataFrame['sellside_con_ent'],
               'diff_ent': DataFrame['buyside_con_ent'] - DataFrame['sellside_con_ent'],
               'sum_ent': DataFrame['buyside_con_ent'] + DataFrame['sellside_con_ent']}

        self.save_image.multi_images(dic)

env = Environments()
env.run3()
