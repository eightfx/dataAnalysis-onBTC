import numpy as np
import pandas as pd
from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

import os
import matplotlib.pyplot as plt

from scipy.fftpack import fft
from modules.loads import Load_data, Reshape_data, Discord_system
from lib.discord.send_toDiscord import Send_toDiscord
from lib.utils.reshape_data import Reshape_data
from lib.mysql.load_data import Load_data

import matplotlib.animation as animation


class Environments:
    def __init__(self):

        self.Load_data = Load_data()
        self.Reshape_data = Reshape_data()
        self.Send_toDiscord= Send_toDiscord()

    def get_del_data(self,DataFrame):
        buyvol = DataFrame[DataFrame['direction'] == 'BUY']['vol']
        sellvol = DataFrame[DataFrame['direction'] == 'SELL']['vol']
        if len(buyvol) > len(sellvol):
            buyvol = buyvol.iloc[1:]
        elif len(sellvol) > len(buyvol):
            sellvol = sellvol.iloc[1:]

        delvol = np.array(buyvol) - np.array(sellvol)
        return delvol

    def run(self):
        image_path = root_path / "data" / "image" / "test.png"
        DataFrame = self.Load_data.read_from_sql('tickdata_gmo',700000) 
        DataFrame = self.Reshape_data.groupby_switch(DataFrame)

        data = self.get_del_data(DataFrame)[10000:]
        print(len(data))

        lim = 100
        images = []
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
 
        for i in range(lim):
            limdata = data[i*100:i*100+500]
            L = len(limdata)
            sf = 200
            # フーリエ変換
            freq = np.linspace(0, sf, L) # 周波数領域の横軸
            yf = np.abs(fft(limdata)) # 振幅スペクトル
            yf = yf * yf # パワースペクトル

            # 周波数領域の描画
            image = ax.plot(freq[0:int(L/2)], yf[0:int(L/2)],color='blue')
            images.append(image)

        anime = animation.ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=0)
        anime.save(image_path)
        self.Discord_system.push_image(image_path)
env = Environments()
env.run()
