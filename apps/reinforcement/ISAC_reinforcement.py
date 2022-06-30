import datetime
import json
import pandas as pd
import traceback

from scipy import optimize
from itertools import chain
import numpy as np
import math
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple
import scipy
from scipy.stats import norm, pearsonr, spearmanr, ttest_1samp, ttest_ind


BATCH_SIZE = 32
CAPACITY = 10000
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))
GAMMA = 0.99  # 時間割引率
NUM_EPISODES = 300  # 最大試行回数

class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_mid)
        self.fc4 = nn.Linear(n_mid, n_mid)
        self.fc5 = nn.Linear(n_mid, n_mid)
        self.fc6 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        output = self.fc6(h5)
        return output

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # CartPoleの行動（右に左に押す）の2を取得

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory(CAPACITY)

        # ニューラルネットワークを構築
        n_in, n_mid, n_out = num_states,200, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)  # Netクラスを使用
        self.target_q_network = Net(n_in, n_mid, n_out)  # Netクラスを使用

        # 最適化手法の設定
        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.0001)

    def replay(self):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        # 1. メモリサイズの確認
        if len(self.memory) < BATCH_SIZE:
            return

        # 2. ミニバッチの作成
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 3. 教師信号となるQ(s_t, a_t)値を求める
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4. 結合パラメータの更新
        self.update_main_q_network()

    def decide_action(self, state, episode):
        '''現在の状態に応じて、行動を決定する'''
        # ε-greedy法で徐々に最適行動のみを採用する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
            # ネットワークの出力の最大値のindexを取り出します = max(1)[1]
            # .view(1,1)は[torch.LongTensor of size 1]　を size 1x1 に変換します

        else:
            # 0,1の行動をランダムに返す
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 0,1の行動をランダムに返す
            # actionは[torch.LongTensor of size 1x1]の形になります

        return action

    def make_minibatch(self):
        '''2. ミニバッチの作成'''

        # 2.1 メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 各変数をミニバッチに対応する形に変形
        # transitionsは1stepごとの(state, action, state_next, reward)が、BATCH_SIZE分格納されている
        # つまり、(state, action, state_next, reward)×BATCH_SIZE
        # これをミニバッチにしたい。つまり
        # (state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)にする
        batch = Transition(*zip(*transitions))

        # 2.3 各変数の要素をミニバッチに対応する形に変形し、ネットワークで扱えるようVariableにする
        # 例えばstateの場合、[torch.FloatTensor of size 1x4]がBATCH_SIZE分並んでいるのですが、
        # それを torch.FloatTensor of size BATCH_SIZEx4 に変換します
        # 状態、行動、報酬、non_finalの状態のミニバッチのVariableを作成
        # catはConcatenates（結合）のことです。
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])


        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        '''3. 教師信号となるQ(s_t, a_t)値を求める'''

        # 3.1 ネットワークを推論モードに切り替える
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 ネットワークが出力したQ(s_t, a_t)を求める
        # self.model(state_batch)は、右左の両方のQ値を出力しており
        # [torch.FloatTensor of size BATCH_SIZEx2]になっている。
        # ここから実行したアクションa_tに対応するQ値を求めるため、action_batchで行った行動a_tが右か左かのindexを求め
        # それに対応するQ値をgatherでひっぱり出す。
        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        # 3.3 max{Q(s_t+1, a)}値を求める。ただし次の状態があるかに注意。

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)),dtype=torch.bool)
        # まずは全部0にしておく
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 次の状態での最大Q値の行動a_mをMain Q-Networkから求める
        # 最後の[1]で行動に対応したindexが返る
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        # 次の状態があるものだけにフィルターし、size 32を32×1へ
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 次の状態があるindexの、行動a_mのQ値をtarget Q-Networkから求める
        # detach()で取り出す
        # squeeze()でsize[minibatch×1]を[minibatch]に。
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        '''4. 結合パラメータの更新'''

        # 4.1 ネットワークを訓練モードに切り替える
        self.main_q_network.train()

        # 4.2 損失関数を計算する（smooth_l1_lossはHuberloss）
        # expected_state_action_valuesは
        # sizeが[minbatch]になっているので、unsqueezeで[minibatch x 1]へ
        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        # 4.3 結合パラメータを更新する
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新

    def update_target_q_network(self):  # DDQNで追加
        '''Target Q-NetworkをMainと同じにする'''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


class Agent:
    def __init__(self, num_states, num_actions):
        '''課題の状態と行動の数を設定する'''
        self.brain = Brain(num_states, num_actions)  # エージェントが行動を決定するための頭脳を生成

    def update_q_function(self):
        '''Q関数を更新する'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''行動を決定する'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        '''Target Q-NetworkをMain Q-Networkと同じに更新'''
        self.brain.update_target_q_network()

class ISAC:
    def __init__(self):
        self.max_lot = 0.03
        self.buyside_gamma = 0
        self.sellside_gamma = 0
        self.sigma = 0
        self.T = 300
        self.count = 0
        self.buyside_alpha = 0
        self.sellside_alpha = 0

    def get_quote(self, buyside_gamma, sellside_gamma):
        if buyside_gamma == 0:
            buyside_gamma = 0.01
        if sellside_gamma == 0:
            sellside_gamma = 0.01

        self.count += 1

        buyside_amount = buyside_gamma * (self.sigma ** 2) * (self.T - self.count) + (2 / buyside_gamma) +math.log(1 + (buyside_gamma / self.buyside_alpha))
        sellside_amount = sellside_gamma * (self.sigma ** 2) * (self.T - self.count) + (2 / sellside_gamma) +math.log(1 + (sellside_gamma / self.sellside_alpha))
        return buyside_amount, sellside_amount

    def calc_sigma(self,DataFrame):
        self.sigma = DataFrame['vol'].std()

    def refresh_params(self,DataFrame):
        self.calc_alpha(DataFrame)
        self.calc_sigma(DataFrame)

    def calc_alpha(self, DataFrame_):
        DataFrame= (DataFrame_['buy_vol'].dropna() *100).astype('int')
        try:
            buyside_params = self.__get_params(DataFrame)
        except:
            pass

        DataFrame= (DataFrame_['sell_vol'].dropna() *100).astype('int')
        try:
            sellside_params = self.__get_params(DataFrame)
        except:
            pass

        self.buyside_alpha, self.sellside_alpha = buyside_params[1], sellside_params[1]

    def __function(self,t,a,b):
        return a*np.exp(-b*t)

    def __get_params(self,DataFrame):
        x_b,y_b = self.__from_DataFrame_to_trainnp(DataFrame)
        buyside_params = self.__curve_fitting(x_b,y_b)
        return buyside_params

    def __from_DataFrame_to_trainnp(self,df):

        df_hist = df.value_counts().sort_index()
        x = np.array(df_hist.index)

        df_hist = np.array(df_hist)
        df_hist = df_hist/df_hist.sum()
        y = np.array(df_hist)

        return x,y

    def __curve_fitting(self,x,y):
        params, params_cov = optimize.curve_fit(self.__function, x, y)
        return params



class Environment:
    def __init__(self):
        self.length = 5000
        self.num_states = 20
        self.num_actions =2
        self.agent = Agent(self.num_states,self.num_actions)
        self.entry_price = 0
        self.stock = 0
        self.balance = 0
        self.count = 0
        self.direction = None
        self.balance_list = []
        self.isac = ISAC()
        self.now_data = None

    def get_states(self,DataFrame, step):
        self.now_data = DataFrame.iloc[step: step + self.num_states]
        state = np.array(self.now_data['vol'].iloc[-self.num_states:])
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0)  # size 4をsize 1x4に変換
        return state


        
    def run(self):
        DataFrame = np.load('time_gmo.npy')
        DataFrame = pd.DataFrame({'buy_vol':DataFrame[:,0], 'sell_vol':DataFrame[:,1],
                                  'sigma':DataFrame[:,2],
                                  'buyside_alpha':DataFrame[:,3],
                                  'sellside_alpha':DataFrame[:,4]})
        DataFrame['cum_vol'] = DataFrame['buy_vol'] - DataFrame['sell_vol']
        DataFrame['vol'] = DataFrame['buy_vol'] + DataFrame['sell_vol']
        DataFrame['cum_vol'] = DataFrame['cum_vol'].cumsum()


        for episode in range(NUM_EPISODES):
            self.balance_list = []
            self.balance = 0
            self.count = 0

            for step in range(len(DataFrame) - self.num_states):
                done = False
                state = self.get_states(DataFrame,step)
                action = self.agent.get_action(state,episode)
                buyside_gamma = int(action[0][0])
                sellside_gamma = int(action[0][0])

                self.isac.sigma = DataFrame['sigma'].iloc[step]
                self.isac.buyside_alpha = DataFrame['buyside_alpha'].iloc[step]
                self.isac.sellside_alpha = DataFrame['sellside_alpha'].iloc[step]
                buyside_amount, sellside_amount = self.isac.get_quote(buyside_gamma,sellside_gamma)

                buyside_done = False
                sellside_done = False
                self.stock = 0
                self.entry_price = 0
                reward = torch.FloatTensor([0.0])
                for j in range(step + 1,len(DataFrame) - self.num_states):
                    if buyside_amount < DataFrame['buy_vol'].iloc[step+self.length] and not buyside_done:
                        self.entry_price -= DataFrame['cum_vol'].iloc[step+self.length] - buyside_amount
                        buyside_done = True

                    if sellside_amount < DataFrame['sell_vol'].iloc[step+self.length] and not sellside_done:
                        self.entry_price +=DataFrame['cum_vol'].iloc[step+self.length] + sellside_amount
                        sellside_done = True

                    if buyside_done and sellside_done:
                        done = True
                        reward = torch.FloatTensor([self.entry_price])
                        break
                    elif (j - step) > self.isac.T:
                        done = True
                        reward = torch.FloatTensor([-1.0])
                        break


                print(reward)
                state_next = None
                # メモリに経験を追加
                self.agent.memorize(state, action, state_next, reward)
                self.balance_list.append(float(list(reward)[0]))

                # Experience ReplayでQ関数を更新する
                self.agent.update_q_function()


            # DDQNで追加、2試行に1度、Target Q-NetworkをMainと同じにコピーする
            if(episode % 2 == 0):
                self.agent.update_target_q_function()

            print(sum(self.balance_list))
            torch.save(self.agent.brain.main_q_network.state_dict(), "reinforcement_model.pth")

if __name__ == '__main__':
    env = Environment()
    env.run()

