from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

import os
from lib.mysql.setting import *
from pathlib import Path
import datetime
import pandas as pd

class Load_data:
    """
    MySQLからデータを読み込むclass
    """
    def __init__(self):
        pass

    def read_from_sql(self,params_name,limit):
        """
        約定履歴を読み込む
        """
        data = pd.read_sql_query(sql='SELECT * FROM {} ORDER BY id DESC LIMIT {}'.format(params_name,limit), con=ENGINE)[::-1].reset_index(drop = True)
        if 'ftx' in params_name:
            data['timestamp'] = data['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x) )
            data['direction'] = data['direction'].replace('buy', 'BUY').replace('sell', 'SELL').replace('Buy',"BUY").replace('Sell', 'SELL')
            data = data.rename(columns={'timestamp':'time'})
        elif 'deribit' in params_name:

            data['direction'] = data['direction'].replace('buy', 'BUY').replace('sell', 'SELL')
        else:
            data['timestamp'] = data['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x/1000) )
            data = data.rename(columns={'timestamp':'time'})
        #b = json.loads(json.loads(data['buyside_board'][10]))
        return data

    def read_from_sql_by_time(self,params_name,start_date,end_date):
        """
        時間指定で約定履歴を読み込む
        """
        start_timestamp = int(start_date.timestamp()*1000)
        end_timestamp = int(end_date.timestamp()*1000)

        data = pd.read_sql_query(sql='SELECT * FROM {} WHERE {} < timestamp and timestamp < {}'.format(params_name,start_timestamp,end_timestamp), con=ENGINE)
        data['timestamp'] = data['timestamp'].map(lambda x: datetime.datetime.fromtimestamp(x/1000) )
        data = data.rename(columns={'timestamp':'time'})
        #b = json.loads(json.loads(data['buyside_board'][10]))
        return data

