from pathlib import Path
import discord
import sys
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from dotenv import load_dotenv
load_dotenv()
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

class Send_toDiscord:
    def __init__(self):
        self.TOKEN = os.environ["DISCORD_TOKEN"]
        self.CHANNEL_ID = os.environ["DISCORD_CHANNEL_ID"]

    def push_message(self,content):
        """
        メッセージをdiscordに送信する
        """
        TOKEN =self.TOKEN
        CHANNEL_ID = self.CHANNEL_ID
       # 接続に必要なオブジェクトを生成
        client = discord.Client()
        # 起動時に動作する処理
        @client.event
        async def on_ready():
            # 起動したらターミナルにログイン通知が表示される
            channel_sent = client.get_channel(CHANNEL_ID)
            await channel_sent.send(content)
            os._exit(1)

        client.run(TOKEN)

    def push_scatter(self,x,y, start,end,span,png_path):
        """
        データの散布図をdiscordに送信する
        """
        xy = np.vstack([x,y])
        cor = np.corrcoef(xy)[0][1]
        plt.title(cor)
        plt.scatter(x,y, s= 5)
        for i in range(int((end - start) / span)):
            x_ = x[(start + i*span< x) & (x < start + (i+1) * span)]
            mean_y = y[(start + i*span< x) & (x < start + (i+1) * span)].mean()
            y_ = [mean_y for i in range(len(x_))]
            print("{}_{}:{}".format(start + i*span,start + (i+ 1)*span, mean_y))
            plt.plot(x_,y_,color='r')
        plt.savefig(png_path)
        self.push_image(png_path)


    def push_image(self,file_path):
        """
        画像ファイルのパスを入力するとその画像をdiscordに送信する
        """
        TOKEN =self.TOKEN
        CHANNEL_ID = self.CHANNEL_ID
       # 接続に必要なオブジェクトを生成
        client = discord.Client()
        # 起動時に動作する処理
        @client.event
        async def on_ready():
            # 起動したらターミナルにログイン通知が表示される
            channel_sent = client.get_channel(CHANNEL_ID)
            await channel_sent.send(file=discord.File(file_path))
            os._exit(1)


        client.run(TOKEN)

    def push_candle(self,DataFrame,image_path):
        """
        約定履歴をローソク足に変換してdiscordに送信する
        """
        if 'vol' in DataFrame.columns:
            DataFrame = DataFrame[['op','hi','lo','cl','vol']].rename(columns={'op':'Open','hi':'High','lo':'Low','cl':'Close','vol':'Volume'})
        else:
            DataFrame = DataFrame[['op','hi','lo','cl']].rename(columns={'op':'Open','hi':'High','lo':'Low','cl':'Close'})
        DataFrame.index = DataFrame.index.map(lambda x: datetime.datetime.fromtimestamp(x))
       
        mpf.plot(DataFrame, type='candle', savefig=image_path)
        self.push_image(image_path)

