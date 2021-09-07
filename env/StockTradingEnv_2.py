# coding=utf-8
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import pdb
import datetime
import os
import math

INITIAL_ACCOUNT_BALANCE = 3000
MAX_ACCOUNT_BALANCE = 10000
MAX_Shape = 10
MIN_Shape = -3
MAX_Coin = 10
fea_num = 21


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        self.action_space = spaces.Box(low=np.array(
            [-0.2, 0.1, 1]), high=np.array([0.2, 0.6, 10]), dtype=np.float16)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(fea_num, 1), dtype=np.float16)

        self.quote = INITIAL_ACCOUNT_BALANCE
        self.base = 0
        self.times = 0
        self.sharp = 0
        self.profit = []
        self.btcOrderBook = df
        self.btcOrderBook.iloc[:, 0] = self.btcOrderBook.iloc[:, 0] / 1000
        self.btcOrderBook.iloc[:, 0] = self.btcOrderBook.iloc[:, 0].apply(
            datetime.datetime.utcfromtimestamp)
        self.btcOrderBook.set_index(self.btcOrderBook.columns[0], inplace=True)
        self.act = None
        lr = np.log(self.btcOrderBook['wp']).diff(1)
        bpd1 = np.log(
            self.btcOrderBook['bp1'] *
            self.btcOrderBook['bv1']).diff(1)
        bpd2 = np.log(
            self.btcOrderBook['bp2'] *
            self.btcOrderBook['bv2']).diff(1)
        bpd3 = np.log(
            self.btcOrderBook['bp3'] *
            self.btcOrderBook['bv3']).diff(1)
        bpd4 = np.log(
            self.btcOrderBook['bp4'] *
            self.btcOrderBook['bv4']).diff(1)
        bpd5 = np.log(
            self.btcOrderBook['bp5'] *
            self.btcOrderBook['bv5']).diff(1)
        apd1 = np.log(
            self.btcOrderBook['ap1'] *
            self.btcOrderBook['av1']).diff(1)
        apd2 = np.log(
            self.btcOrderBook['ap2'] *
            self.btcOrderBook['av2']).diff(1)
        apd3 = np.log(
            self.btcOrderBook['ap3'] *
            self.btcOrderBook['av3']).diff(1)
        apd4 = np.log(
            self.btcOrderBook['ap4'] *
            self.btcOrderBook['av4']).diff(1)
        apd5 = np.log(
            self.btcOrderBook['ap5'] *
            self.btcOrderBook['av5']).diff(1)
        s1 = (self.btcOrderBook['ap1'] - self.btcOrderBook['bp1']) / \
             (self.btcOrderBook['ap1'] + self.btcOrderBook['bp1'])
        s2 = (self.btcOrderBook['ap2'] - self.btcOrderBook['bp2']) / \
             (self.btcOrderBook['ap2'] + self.btcOrderBook['bp2'])
        s3 = (self.btcOrderBook['ap3'] - self.btcOrderBook['bp3']) / \
             (self.btcOrderBook['ap3'] + self.btcOrderBook['bp3'])
        s4 = (self.btcOrderBook['ap4'] - self.btcOrderBook['bp4']) / \
             (self.btcOrderBook['ap4'] + self.btcOrderBook['bp4'])
        s5 = (self.btcOrderBook['ap5'] - self.btcOrderBook['bp5']) / \
             (self.btcOrderBook['ap5'] + self.btcOrderBook['bp5'])
        cf5 = self.btcOrderBook['cf'].rolling(5).sum()
        cf3 = self.btcOrderBook['cf'].rolling(3).sum()
        lr = self.scale(lr, 5)
        bpd1 = self.scale(bpd1, 10)
        bpd2 = self.scale(bpd2, 10)
        bpd3 = self.scale(bpd3, 10)
        bpd4 = self.scale(bpd4, 10)
        bpd5 = self.scale(bpd5, 10)
        apd1 = self.scale(apd1, 10)
        apd2 = self.scale(apd2, 10)
        apd3 = self.scale(apd3, 10)
        apd4 = self.scale(apd4, 10)
        apd5 = self.scale(apd5, 10)
        s1 = self.scale(s1, 10)
        s2 = self.scale(s2, 10)
        s3 = self.scale(s3, 10)
        s4 = self.scale(s4, 10)
        s5 = self.scale(s5, 10)
        cf = self.scale(self.btcOrderBook['cf'], 10)
        cf5 = self.scale(cf5, 10)
        cf3 = self.scale(cf3, 10)
        self.fea = pd.DataFrame([lr,
                                 bpd1,
                                 bpd2,
                                 bpd3,
                                 bpd4,
                                 bpd5,
                                 apd1,
                                 apd2,
                                 apd3,
                                 apd4,
                                 apd5,
                                 s1,
                                 s2,
                                 s3,
                                 s4,
                                 s5,
                                 cf,
                                 cf5,
                                 cf3],
                                index=['lr',
                                       'bpd1',
                                       'bpd2',
                                       'bpd3',
                                       'bpd4',
                                       'bpd5',
                                       'apd1',
                                       'apd2',
                                       'apd3',
                                       'apd4',
                                       'apd5',
                                       's1',
                                       's2',
                                       's3',
                                       's4',
                                       's5',
                                       'cf',
                                       'cf5',
                                       'cf3']).T.dropna()
        self.btcOrderBook = self.btcOrderBook.loc[self.fea.index, :]
        print('load data done')

    def readfile(self, path):
        files = os.listdir(path)
        file_list = []
        for _file in files:  # 遍历文件夹
            if not os.path.isdir(_file):
                if "csv" in _file:
                    file_list.append(_file)
        return file_list

    def scale(self, series, d):
        series = series.dropna()
        return (series - series.rolling(d).min()) / \
            (series.rolling(d).max() - series.rolling(d).min())

    def getobs(self, curIdx):
        obs = np.r_[np.array(self.fea.iloc[curIdx, :]),
                    np.array([self.quote / MAX_ACCOUNT_BALANCE,
                              self.base / MAX_Coin])].reshape(-1, 1)
        # print(obs)
        return obs

    def step(self, action):
        volume, dw, cdSec = action  # volume 量  dw ？ cdsec 持仓时间
        self.act = action
        cdSec = float(cdSec)
        dt = self.btcOrderBook.iloc[self.curIdx, :].name
        bp = self.btcOrderBook.iloc[self.curIdx, :10]
        bv = self.btcOrderBook.iloc[self.curIdx, 10:20]
        ap = self.btcOrderBook.iloc[self.curIdx, 20:30]
        av = self.btcOrderBook.iloc[self.curIdx, 30:40]
        highp = self.btcOrderBook['h'].loc[dt +
                                           datetime.timedelta(seconds=1.0):dt +
                                           datetime.timedelta(seconds=cdSec)]
        lowp = self.btcOrderBook['l'].loc[dt +
                                          datetime.timedelta(seconds=1.0):dt +
                                          datetime.timedelta(seconds=cdSec)]
        this_index = self.curIdx
        if volume > 0 and self.quote >= volume * bp[0]:  # 买
            try:
                bidOrderPrice = np.array(bp)[bv.cumsum() > dw][0]
            except BaseException:
                # print(np.array(bp))
                # print("*********")
                # print(np.array(bv))
                # print("*********")
                # print(bv.cumsum())
                # print("*********")
                print("dw error", dw)
                bidOrderPrice = None
            if bidOrderPrice and len(
                    lowp[lowp <= bidOrderPrice]) > 0:  # 未来一段时间 如果有个最低价小于目前的出价 就买入
                self.quote = self.quote - volume * bidOrderPrice
                self.base = self.base + volume
                self.curIdx = len(self.btcOrderBook[self.btcOrderBook.index[0]:lowp[lowp <= bidOrderPrice].index[
                    0]])   # 这里是到最低价的index + 1
            else:
                self.curIdx = self.curIdx + math.ceil(cdSec)
        elif volume < 0 and self.base > -volume:  # 卖
            try:
                askOrderPrice = np.array(ap)[av.cumsum() > dw][0]
            except BaseException:
                # print(np.array(ap))
                # print("*********")
                # print(np.array(av))
                # print("*********")
                # print(av.cumsum())
                # print("*********")
                print("dw error", dw)
                askOrderPrice = None
            if askOrderPrice and len(highp[highp >= askOrderPrice]) > 0:
                self.quote = self.quote - volume * askOrderPrice
                self.base = self.base + volume
                self.curIdx = len(
                    self.btcOrderBook[self.btcOrderBook.index[0]:highp[highp >= askOrderPrice].index[0]])
            else:
                self.curIdx = self.curIdx + math.ceil(cdSec)
        elif volume == 0:
            self.curIdx = self.curIdx + math.ceil(cdSec)
        if this_index == self.curIdx:
            self.curIdx += 1
        profit = self.quote + self.base * \
            self.btcOrderBook['wp'].iloc[self.curIdx] - INITIAL_ACCOUNT_BALANCE
        self.profit.append(profit)
        if len(self.profit) > 10 and np.std(np.array(self.profit)) != 0:
            reward = np.mean(np.array(self.profit)) / \
                np.std(np.array(self.profit))
        else:
            reward = 0
        obs = self.getobs(self.curIdx)
        done = reward <= MIN_Shape or self.curIdx > (
            len(self.fea) - 30) or reward > MAX_Shape
        if not done:
            self.sharp = reward
        # done = (reward>2 and  self.curIdx>10000) or  self.curIdx > (len(self.fea) - 10)
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        # try:
        #     self.curIdx > 0
        # except:
        # files = self.readfile("./data/")
        # _file = random.choice(files)
        # print("选择文件{}".format(_file))
        # df = pd.read_csv("./data/{}".format(_file))
        # self.reset_df(df)
        self.quote = INITIAL_ACCOUNT_BALANCE
        self.base = 0
        # self.curIdx = random.randint(0, 10)
        self.curIdx = 0
        obs = self.getobs(self.curIdx)
        if len(self.profit) > 0:
            self.times = self.times + 1
            print('----第',
                  self.times,
                  '次开始，上次的tick级夏普是,',
                  self.sharp,
                  '成交量:',
                  len(self.profit),
                  'profits sum',
                  np.sum(self.profit),
                  'total profit:',
                  self.profit[-1],
                  '----')
        self.sharp = 0
        self.profit = []
        return obs

    def render(self, mode='human', close=False):
        currentPrice = self.btcOrderBook['wp'].iloc[self.curIdx]
        netWorth = self.quote + self.base * \
            self.btcOrderBook['wp'].iloc[self.curIdx]
        profit = netWorth - INITIAL_ACCOUNT_BALANCE
        sharp = self.sharp
        curIdx = self.curIdx
        times = self.times
        print(
            '-----------------------',
            'times: {}'.format(times),
            '--------------------------')
        #print("curObs", self.getobs(self.curIdx))
        #print('curAction', list(self.act))
        print('curIdx: {}'.format(curIdx))
        print('currentPrice: {}'.format(currentPrice))
        print("quote: {} coin: {}".format(self.quote, self.base))
        print('netWorth: {}'.format(netWorth))
        print('Profit: {}'.format(profit))
        print('sharp: {}'.format(sharp))
