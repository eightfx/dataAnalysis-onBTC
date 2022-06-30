from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

import pandas as pd
from lib.utils.reshape_data import Reshape_data
from lib.mysql.load_data import Load_data

class BackTest():
    def __init__(self, logic_class):
        self.logic = logic_class
        self.Load_data = Load_data
        self.data_len = self.logic.input_length

        self.__stock = []
        self.__balance_list = [0]

    def run(self, DataFrame):

        for i in range(len(DataFrame) - self.data_len):
            input_data = {'df_raw': DataFrame.iloc[i:i+self.data_len].reset_index(drop=True)}
            output_data = self.logic.main(input_data)
            now_data = DataFrame.iloc[i + self.data_len]

            profit = self.execution(now_data, output_data)
            self.__balance_list.append(self.__balance_list[-1] + profit)
            if i % 1000 == 0:
                print(i)

        return self.__balance_list



    def execution(self, now_data, output_data):
        if len(self.__stock ) == 0:
            if output_data['direction'] == 'BUY':
                self.__stock.append([now_data['price'], 0.01, 1, 0])
            if output_data['direction'] == 'SELL':
                self.__stock.append([now_data['price'], -0.01, 1, 0])

            profit = 0

        else:
            if output_data['should_close_buy'] == True and self.__stock[0][0] > 0:
                profit = now_data['price'] * self.__stock[0][1] - self.__stock[0][0] * self.__stock[0][1]

            elif output_data['should_close_sell'] == True and self.__stock[0][0] < 0:
                profit = now_data['price'] * self.__stock[0][1] - self.__stock[0][0] * self.__stock[0][1]
            else:
                profit = 0

        return profit













        
