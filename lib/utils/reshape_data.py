import json
import pandas as pd
import numpy as np

class Reshape_data:
    def __init__(self):
        pass

    #{'price', 'direction', 'amount'}をcolumnsとする時系列データをOHLCV形式に圧縮する
    def groupby_to_ohlcv(self,df,keys):

        DataFrame = df[[keys,'price']].groupby([keys]).min().rename(columns={'price':'lo'})
        DataFrame['hi'] = df[[keys,'price']].groupby([keys]).max()
        DataFrame['op'] = df[[keys,'price']].groupby([keys]).first()
        DataFrame['cl'] = df[[keys,'price']].groupby([keys]).last()
        #DataFrame['time'] = df[[keys,'price']].groupby([keys]).last()
        if 'amount' in df.columns:
            DataFrame['buy_vol'] = df[[keys,'amount']][df['direction'] == 'BUY'].groupby([keys]).sum()
            DataFrame['sell_vol'] = df[[keys,'amount']][df['direction'] == 'SELL'].groupby([keys]).sum()
            DataFrame['vol'] = df[[keys,'amount']].groupby([keys]).sum()
        if 'buy_ent' in df.columns:
            DataFrame['buy_ent'] = df[[keys,'buy_ent']].groupby([keys]).last()
        if 'sell_ent' in df.columns:
            DataFrame['sell_ent'] = df[[keys,'sell_ent']].groupby([keys]).last()

        return DataFrame

    def from_jsonBoard_to_funcBoard(self,now_data):
        bids = pd.DataFrame(json.loads(json.loads(now_data['buyside_board']))).rename(columns={0:'price',1:'size'}).astype('float').sort_values('price',ascending = False)
        asks = pd.DataFrame(json.loads(json.loads(now_data['sellside_board']))).rename(columns={0:'price',1:'size'}).astype('float').sort_values('price')
        bids_function = np.array(bids['price'].repeat(bids['size']*100))
        asks_function = np.array(asks['price'].repeat(asks['size']*100))

        output = {'bids_function':bids_function,
                  'asks_function':asks_function}
        return output

 
    def groupby_board(self,df_gmo,df_board):

        df_board['data'] = 'board'
        df_gmo['data'] = 'tickdata'

        DataFrame = pd.concat([df_gmo, df_board]).sort_values('time')

        i = 0
        keys = []
        l = []
        for key,grouper in groupby(DataFrame["data"]):
            l.extend([i for k in range(len(list(grouper)))])
            keys.extend([key])
            i = i+1
        DataFrame["index"] = l
        if DataFrame['data'].iloc[0] == 'tickdata':
            DataFrame = DataFrame.iloc[1:]
        print(DataFrame)
 
        df_gmo = DataFrame[DataFrame['data'] == 'tickdata']
        df_board = DataFrame[DataFrame['data'] == 'board']

        df_gmo = self.groupby_to_ohlcv(df_gmo,'index')
        df_board = df_board[['index','buyside_board','sellside_board']].groupby(['index']).first()

        if len(df_gmo) > len(df_board):
            df_gmo = df_gmo.iloc[1:]
        elif len(df_board) > len(df_gmo):
            df_board = df_board.iloc[1:]

        output = {'tick':df_gmo,
                  'board':df_board}
        return output
 

    # 時間軸でまとめてOHLCV化する
    def groupby_time(self,df,resolution:str,span:int):
        
        if resolution == 'day':
            df["time"] = df['time'].map(lambda x: x.replace(day=x.day//span*span,hour=0,minute=0,second=0, microsecond=0))
        if resolution == 'hour':
            df["time"] = df['time'].map(lambda x: x.replace(hour =x.hour//span*span ,minute = 0,second=0, microsecond=0))
        if resolution == 'minute':
            df["time"] = df['time'].map(lambda x: x.replace(minute=x.minute//span*span,second=0, microsecond=0))


        if resolution == 'second':
            df["time"] = df['time'].map(lambda x: x.replace(second=x.second//span*span, microsecond=0))

        if resolution == 'microsecond':
            df["time"] = df['time'].map(lambda x: x.replace(microsecond=x.microsecond//span*span))


        DataFrame = self.groupby_to_ohlcv(df,'time')
        return DataFrame

    # directionが連続するものをひとまとめにOHLCVにする
    def groupby_switch(self,DataFrame):
        i = 0
        keys = []
        l = []
        for key,grouper in groupby(DataFrame["direction"]):
            l.extend([i for k in range(len(list(grouper)))])
            keys.extend([key])
            i = i+1
        DataFrame["index"] = l
        DataFrame = self.groupby_to_ohlcv(DataFrame,'index')

        DataFrame["direction"] = keys
        return DataFrame

