from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))
from lib.mysql.setting import session
# Userモデルの取得
import datetime
import json
import pandas as pd
from itertools import chain
import numpy as np

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



class Load_data:
    def __init__(self):
        pass

    def read_from_sql(self,params_name):
        data = pd.read_sql_query(sql='SELECT * FROM {} ORDER BY id DESC LIMIT 100000'.format(params_name), con=ENGINE)
        #b = json.loads(json.loads(data['buyside_board'][10]))
        return data
# 経験を保存するメモリクラスを定義します

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


class Environment:
    def __init__(self):
        self.num_states = 20
        self.num_actions =20
        self.agent = Agent(self.num_states,self.num_actions)
        self.entry_price = 0
        self.stock = 0
        self.balance = 0
        self.count = 0
        self.direction = None
        self.Load_data = Load_data()
        self.balance_list = []
    def run(self):
        #板情報の取得
        df_board = self.Load_data.read_from_sql('gmo_board')
        df_board["time"] = df_board['time'].map(lambda x: x.replace(second=x.second//1, microsecond=0))
        DataFrame_board =  df_board[['time','buyside_board','sellside_board']].groupby(['time']).first()

        #約定履歴の取得
        df_gmo = self.Load_data.read_from_sql('tickdata_gmo').iloc[::-1]

        df_gmo["time"] = df_gmo['time'].map(lambda x: x.replace(second=x.second//1, microsecond=0))
        DataFrame = df_gmo[['time','price']].groupby(['time']).min().rename(columns={'price':'lo'})
        DataFrame['hi'] = df_gmo[['time','price']].groupby(['time']).max()
        DataFrame['op'] = df_gmo[['time','price']].groupby(['time']).first()
        DataFrame['cl'] = df_gmo[['time','price']].groupby(['time']).last()
        DataFrame['buy_vol'] = df_gmo[['time','amount']][df_gmo['direction'] == 'BUY'].groupby(['time']).sum()
        DataFrame['sell_vol'] = df_gmo[['time','amount']][df_gmo['direction'] == 'SELL'].groupby(['time']).sum()
        Datas = pd.merge(DataFrame,DataFrame_board,on = 'time',how='outer')
        Datas['buy_vol'] = Datas['buy_vol'].fillna(0)
        Datas['sell_vol'] = Datas['sell_vol'].fillna(0)

        Datas = Datas.dropna()


        

        for start in range(10000):
            self.balance_list = []
            for episode in range(NUM_EPISODES):
                self.balance = 0
                self.count = 0

                for step in range(3000+start,3000+start+50):
                    now_data = Datas.iloc[step]
                    now_time = Datas.index[step]
                    done = False

                    df = df_gmo[df_gmo['time'] < now_time]
                    df_buyside = df[df['direction'] == 'BUY']
                    df_sellside = df[df['direction'] == 'SELL']
                    df_buyside = DataFrame['buy_vol'].iloc[step-3000:step]
                    df_sellside = DataFrame['sell_vol'].iloc[step-3000:step]

                    df_hist = df_buyside.value_counts().sort_index()
                    df_hist = df_hist/df_hist.sum()
                    y_b = np.array(df_hist)[:self.num_states]

                    df_hist = df_sellside.value_counts().sort_index()
                    df_hist = df_hist/df_hist.sum()
                    y_s = np.array(df_hist)[:self.num_states]

                    bids = pd.DataFrame(json.loads(json.loads(now_data['buyside_board']))).rename(columns={0:'price',1:'size'}).astype('float').sort_values('price',ascending = False)

                    bids_function = np.array(bids['price'].repeat(bids['size']*100))[:self.num_states]
                    bids_function_diff = abs(bids_function - bids_function[0])
                    buyside_CDF = scipy.stats.zscore(bids_function_diff * y_s)


                    asks = pd.DataFrame(json.loads(json.loads(now_data['sellside_board']))).rename(columns={0:'price',1:'size'}).astype('float').sort_values('price')
                    asks_function = np.array(asks['price'].repeat(asks['size']*100))[:self.num_states]
                    asks_function_diff = abs(asks_function - asks_function[0])

                    sellside_CDF = scipy.stats.zscore(asks_function_diff * y_b)


                    now_price = now_data['op']

                    if self.stock == 0:
                        state = sellside_CDF
                        state[0] = 0
                    elif self.stock == 1:
                        state = sellside_CDF
                        state[0] = 1
                    elif self.stock == -1:
                        state = buyside_CDF
                        state[0] = -1
                    state = torch.from_numpy(state).type(torch.FloatTensor)
                    state = torch.unsqueeze(state, 0)  # size 4をsize 1x4に変換

                    action = self.agent.get_action(state,episode)
                    action_lot = int(action[0][0])

                    buyside_limit_price = bids_function[action_lot]
                    sellside_limit_price = asks_function[action_lot]
                    if self.stock == 0:
                        if now_data['hi'] > sellside_limit_price:
                            self.entry_price = sellside_limit_price - 1
                            self.stock = -1
                            self.count += 1
                            reward = torch.FloatTensor([0.0])
                        else:
                            reward = torch.FloatTensor([0.0])

                    elif self.stock == 1:
                        if now_data['hi'] > sellside_limit_price:
                            profit = ((sellside_limit_price - 1) - self.entry_price) * 0.01 - 2
                            self.balance += profit
                            self.stock = 0
                            done = True
                            if profit > 0:
                                reward = torch.FloatTensor([0])
                            else:
                                reward = torch.FloatTensor([0])
                        else:
                            reward = torch.FloatTensor([0.0])


                    elif self.stock == -1:
                        if now_data['lo'] < buyside_limit_price:
                            profit = -((buyside_limit_price + 1) - self.entry_price) *0.01 - 2
                            self.balance += profit

                            self.stock = 0
                            done = True
                            if profit > 0:
                                reward = torch.FloatTensor([0])
                            else:
                                reward = torch.FloatTensor([0])
                        else:
                            reward = torch.FloatTensor([0.0])


                    if done:
                        state_next = None
                        reward = torch.FloatTensor([self.balance/10])
                    else:

                        next_data = Datas.iloc[step+1]
                        bids = pd.DataFrame(json.loads(json.loads(next_data['buyside_board']))).rename(columns={0:'price',1:'size'}).astype('float').sort_values('price',ascending = False)
                        bids_function = np.array(bids['price'].repeat(bids['size']*100))[:self.num_states]
                        buyside_limit_price = bids_function[action_lot]
                        bids_function_diff = abs(bids_function - bids_function[0])
                        buyside_CDF = scipy.stats.zscore(bids_function_diff * y_s)

                        asks = pd.DataFrame(json.loads(json.loads(next_data['sellside_board']))).rename(columns={0:'price',1:'size'}).astype('float').sort_values('price')
                        asks_function = np.array(asks['price'].repeat(asks['size']*100))[:self.num_states]
                        sellside_limit_price = asks_function[action_lot]
                        asks_function_diff = abs(asks_function - asks_function[0])
                        sellside_CDF = scipy.stats.zscore(asks_function_diff * y_b)

                        if self.stock == 0:
                            state_next = sellside_CDF
                            state_next[0] = 0
                        elif self.stock == 1:
                            state_next = sellside_CDF
                            state_next[0] = 1
                        elif self.stock == -1:
                            state_next = buyside_CDF
                            state_next[0] = -1

                        state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                        state_next = torch.unsqueeze(state_next, 0)  # size 4をsize 1x4に変換


                    # メモリに経験を追加
                    self.agent.memorize(state, action, state_next, reward)

                    # Experience ReplayでQ関数を更新する
                    self.agent.update_q_function()

                    if done:
                        break

                self.balance_list.append(self.balance)
                # DDQNで追加、2試行に1度、Target Q-NetworkをMainと同じにコピーする
                if(episode % 2 == 0):
                    self.agent.update_target_q_function()

            print(self.balance_list)
            torch.save(self.agent.brain.main_q_network.state_dict(), "reinforcement_model.pth")


                        
                
               

    def main(self):
        #板情報の取得
        df_board = self.Load_data.read_from_sql('gmo_board')
        df_board["time"] = df_board['time'].map(lambda x: x.replace(second=x.second//1, microsecond=0))
        DataFrame_board =  df_board[['time','buyside_board','sellside_board']].groupby(['time']).first()

        #約定履歴の取得
        df_gmo = self.Load_data.read_from_sql('tickdata_gmo').iloc[::-1]
        df_gmo["time"] = df_gmo['time'].map(lambda x: x.replace(second=x.second//1, microsecond=0))
        DataFrame = df_gmo[['time','price']].groupby(['time']).min().rename(columns={'price':'lo'})
        DataFrame['hi'] = df_gmo[['time','price']].groupby(['time']).max()
        DataFrame['op'] = df_gmo[['time','price']].groupby(['time']).first()
        DataFrame['cl'] = df_gmo[['time','price']].groupby(['time']).last()

        Datas = pd.merge(DataFrame,DataFrame_board,on = 'time',how='outer')
        Datas = Datas.fillna(method = 'ffill').dropna()

        now_time = Datas.index[100]
        now_data = Datas.iloc[100]
        df = df_gmo[df_gmo['time'] < now_time]
        df_buyside = df[df['direction'] == 'BUY']
        df_sellside = df[df['direction'] == 'SELL']
        df_buyside = df_buyside.iloc[-500:]
        df_sellside = df_sellside.iloc[-500:]

        df_hist = df_buyside['amount'].value_counts().sort_index()
        df_hist = df_hist/df_hist.sum()
        y_b = np.array(df_hist)[:15]
        
        df_hist = df_sellside['amount'].value_counts().sort_index()
        df_hist = df_hist/df_hist.sum()
        y_s = np.array(df_hist)[:15]

        bids = pd.DataFrame(json.loads(json.loads(now_data['buyside_board']))).rename(columns={0:'price',1:'size'}).astype('float').sort_values('price',ascending = False)
        bids_function = np.array(bids['price'].repeat(bids['size']*100))[:15]
        bids_function = abs(bids_function - bids_function[0])
        sellside_CDF = bids_function * y_s


        print(df_buyside)
        print(sellside_CDF)


if __name__ == '__main__':
    env = Environment()
    env.run()

