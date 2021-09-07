import pandas as pd
import numpy as np
from history import History
import datetime
import os
import sys
import matplotlib.pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import SAC
from stable_baselines3 import PPO

FIG_PATH = "G:\\RL\\RL_project\\fig\\SAC\\TEST"
DATA_PATH = "G:\\RL\\RL_project\\data\\shaped\\oneweek"
MODEL_PATH = "G:\\RL\\RL_project\\model\\SAC\\my_model.zip"
MODEL_TYPE = "SAC"


class TestEnv:
    def __init__(self, model: BaseAlgorithm, model_name: str):
        self.model = model
        self.model_name = model_name
        self.cur_index = 0
        self.h = History(20)
        self.INITIAL_ACCOUNT_BALANCE = 3000
        self.MAX_ACCOUNT_BALANCE = 10000
        self.MAX_Shape = 5
        self.MIN_Shape = -3
        self.MAX_Coin = 10
        self.fea_num = 21
        self.my_quote = self.INITIAL_ACCOUNT_BALANCE
        self.my_base = 0
        self.init_quote = self.INITIAL_ACCOUNT_BALANCE
        self.init_base = 0
        self.initNetWorth = 0
        self.profits = []
        self.bid1 = 0
        self.ask1 = 0
        self.bid_ok = False
        self.init_bid = 0
        self.INITIAL_ORDER_ID = 0
        self.my_action = []
        # #######cur_order########
        self.cur_order_id = 0
        self.cur_order_volume = 0
        self.cur_order_price = 0
        self.cur_order_dire = ""
        self.waitStart = -1
        self.waitTime = -1
        #########################

        # for debug
        #########################
        self.create_count = 0
        self.fill_count = 0
        self.buy_count = 0
        self.sell_count = 0
        self.wait_count = 0
        self.cd_secs = []
        self.buy_volume = 0
        self.sell_volume = 0
        self.ssv = []
        self.bbv = []
        self.buy_prices = []
        self.buy_dt = []
        self.sell_prices = []
        self.sell_dt = []
        self.sharps = []
        self.coin_state = []
        self.filled_buy_count = 0
        self.filled_sell_count = 0
        self.filled_buy_volume = 0
        self.filled_sell_volume = 0
        self.fsv = []
        self.fbv = []
        self.filled_buy_prices = []
        self.filled_buy_dt = []
        self.filled_sell_prices = []
        self.filled_sell_dt = []

    def reset_env(self):
        self.h.clear()
        self.bid_ok = False
        self.init_quote = self.my_quote
        self.init_base = self.my_base
        self.cur_index = 0
        self.profits = []
        self.my_action = []
        self.create_count = 0
        self.fill_count = 0
        self.buy_count = 0
        self.sell_count = 0
        self.wait_count = 0
        self.cd_secs = []
        self.buy_volume = 0
        self.sell_volume = 0
        self.ssv = []
        self.bbv = []
        self.buy_prices = []
        self.buy_dt = []
        self.sell_prices = []
        self.sell_dt = []
        self.sharps = []
        self.coin_state = []
        self.filled_buy_count = 0
        self.filled_sell_count = 0
        self.filled_buy_volume = 0
        self.filled_sell_volume = 0
        self.fsv = []
        self.fbv = []
        self.filled_buy_prices = []
        self.filled_buy_dt = []
        self.filled_sell_prices = []
        self.filled_sell_dt = []
        self.initNetWorth = 0
        self.bid1 = 0
        self.ask1 = 0
        self.bid_ok = False
        self.init_bid = 0
        self.INITIAL_ORDER_ID = 0

    def generate_order_id(self):
        self.INITIAL_ORDER_ID = self.INITIAL_ORDER_ID + 1
        return self.INITIAL_ORDER_ID

    def create_order(self, d, p, v, i, ws, wt):
        self.create_count += 1
        self.cur_order_id = i
        self.cur_order_volume = v
        self.cur_order_price = p
        self.cur_order_dire = d
        self.waitStart = ws
        self.waitTime = wt
        # print("create order id:", self.create_count)

    def revoke_order(self):

        # print("revoke order id:", self.cur_order_id)
        # print("__________________________________")
        self.cur_order_id = 0
        self.cur_order_volume = 0
        self.cur_order_price = 0
        self.cur_order_dire = ""

    def check_fill(self):

        if not self.cur_order_id == 0:

            if self.cur_order_dire == "buy" and self.cur_order_price >= self.h.history[-1][-1]:
                self.fill_order()

            elif self.cur_order_dire == "sell" and self.cur_order_price <= self.h.history[-1][-2]:
                self.fill_order()

    def fill_order(self):
        #print("fill order id:", self.cur_order_id)

        self.fill_count += 1
        print("第{}次成交:".format(self.fill_count), end=" ")
        if self.cur_order_dire == "buy":
            self.my_quote -= self.cur_order_price * self.cur_order_volume
            self.my_base += self.cur_order_volume
            self.filled_buy_prices.append(self.cur_order_price)
            self.filled_buy_dt.append(self.waitStart)
            self.filled_buy_count += 1
            self.filled_buy_volume += self.cur_order_volume
            self.fbv.append(self.cur_order_volume)
            self.coin_state.append(self.my_base)
        else:
            self.my_quote += self.cur_order_price * self.cur_order_volume
            self.my_base -= self.cur_order_volume
            self.filled_sell_prices.append(self.cur_order_price)
            self.filled_sell_dt.append(self.waitStart)
            self.filled_sell_count += 1
            self.filled_sell_volume += self.cur_order_volume
            self.fsv.append(self.cur_order_volume)
            self.coin_state.append(self.my_base)
        self.revoke_order()
        self.show_profit()

    def show_profit(self):

        if self.initNetWorth == 0:
            self.initNetWorth = self.my_base * self.bid1 + self.my_quote
            print("\tnet_worth ratio: 1 money changed 0 price change sharp 0")
        else:
            self.net_worth = self.my_base * self.bid1 + self.my_quote
            ratio = self.net_worth / self.initNetWorth
            profit = (ratio - 1) * self.initNetWorth
            self.profits.append(profit)
            changed_money = round(profit, 4)
            reward = 0
            if len(self.profits) > 10:
                reward = np.mean(np.array(self.profits)) / \
                    np.std(np.array(self.profits))
            self.sharps.append(reward)
            # print("fill cur_index:", self.cur_index - 19)
            # print("quote:", self.my_quote, " base:", self.my_base)
            # print("net_worth:", self.net_worth)
            print("\tnet_worth ratio: {} money changed {} price change{} sharp {}".format(
                round(ratio, 5), changed_money, round(self.bid1 - self.init_bid, 3), reward))
            # print()

    def get_order_info(self):
        return "id:{} p:{} v:{} d:{} ws:{} wt:{}".format(
            self.cur_order_id,
            self.cur_order_price,
            self.cur_order_volume,
            self.cur_order_dire,
            self.waitStart,
            self.waitTime)

    def step(self):
        self.check_fill()
        cur_data = self.h.history[-1]
        dt = cur_data[0]
        now = datetime.datetime.utcfromtimestamp(dt / 1000)

        if not self.waitStart == -1 and (now - self.waitStart) < self.waitTime:
            return
        if not self.cur_order_id == 0:
            self.revoke_order()

        if self.h.is_full():
            obs = self.h.get_obs()

            a = self.my_quote / self.MAX_ACCOUNT_BALANCE
            b = self.my_base / self.MAX_Coin
            obs = np.append(obs, [a, b]).reshape(self.fea_num, 1)
            #         print(obs)
            if [True] in np.isnan(obs):
                # print("nan obs")
                return
            action = self.model.predict(obs, deterministic=True)
            volume, dw, cd_sec = tuple(action[0])
            my_action = action
            # print("cur_action:",my_action)
            cd_sec = float(cd_sec)

            bp = cur_data[1:11]
            bv = cur_data[11:21]
            ap = cur_data[21:31]
            av = cur_data[31:41]

            if volume > 0 and self.my_quote >= volume * bp[0]:  # 买
                try:
                    bid_order_price = bp[bv.cumsum() > dw][0]
                except BaseException:
                    print("dw error")
                    return
                my_order_id = self.generate_order_id()
                direction = 'buy'
                # print("Action:BUY Price:{},Volume:{},CD:{}".format(bidOrderPrice,volume,cdSec))
                wt = datetime.timedelta(seconds=cd_sec)
                ws = datetime.datetime.utcfromtimestamp(dt / 1000)
                self.create_order(direction, bid_order_price,
                                  volume, my_order_id, ws, wt)
                self.buy_count += 1
                self.buy_volume += volume
                self.buy_prices.append(bid_order_price)
                self.bbv.append(volume)
                self.buy_dt.append(ws)
                self.cd_secs.append(cd_sec)
                # print("cur_dt:",ws)
                # print("cur_obs:",obs)
                # print("create order index:", self.cur_index - 19)
                # print("create order id:",myOrderid)
                # print("cur_quote:",my_quote," cur_base:",my_base)
                # print("bidOrderPrice:",bidOrderPrice)

            elif volume < 0 and self.my_base >= -volume:  # 卖
                try:
                    ask_order_price = ap[av.cumsum() > dw][0]
                except BaseException:
                    print("dw error")
                    return
                my_order_id = self.generate_order_id()
                direction = 'sell'
                wt = datetime.timedelta(seconds=cd_sec)
                ws = datetime.datetime.utcfromtimestamp(dt / 1000)
                self.create_order(
                    direction, ask_order_price, -volume, my_order_id, ws, wt)
                self.sell_count += 1
                self.sell_volume += -volume
                self.ssv.append(-volume)
                self.sell_prices.append(ask_order_price)
                self.sell_dt.append(ws)
                self.cd_secs.append(cd_sec)
                # print("cur_dt:",ws)
                # #print("cur_obs:",obs)
                # print("create order index:",cur_index-19)
                #print("create order id:", ask_order_price)
                # print("cur_quote:",my_quote," cur_base:",my_base)
                # print("askOrderPrice:",askOrderPrice)
            elif volume == 0:
                self.waitTime = datetime.timedelta(seconds=cd_sec)
                self.waitStart = datetime.datetime.utcfromtimestamp(dt / 1000)
                self.wait_count += 1

    def test(self, file: str):
        global DATA_PATH
        df = pd.read_csv(DATA_PATH + "\\" + file)
        length = len(df)
        date = file[-12:-4]
        print("+++++++++++++++++++++++++++++++++++++++数据日期：{}++++++++++++++++++++++++++++++++".format(date))

        for i in range(length):
            self.cur_index = i
            self.h.append(df.loc[self.cur_index].tolist())
            self.bid1 = self.h.history[-1][1]
            self.ask1 = self.h.history[-1][21]
            # print(h.getObs())
            if self.bid1 != 0 and not self.bid_ok:
                self.init_bid = self.bid1
                self.bid_ok = True
            self.step()
        self.save_fig(df, date)
        print("date:{} init quote:{} init base:{} quote:{} base:{} profit:{} sharp{}".format(
            date, self.init_quote, self.init_base, self.my_quote, self.my_base, self.profits[-1], self.sharps[-1]))
        del df
        self.reset_env()

    def save_fig(self, df: pd.DataFrame, date: str):
        wp = df["wp"]
        dt = df["ts"]
        sell_prices = np.array(self.sell_prices)
        buy_prices = np.array(self.buy_prices)
        ssv = np.array(self.ssv)
        bbv = np.array(self.bbv)
        filled_sell_prices = np.array(self.filled_sell_prices)
        filled_buy_prices = np.array(self.filled_buy_prices)
        fbv = np.array(self.fbv)
        fsv = np.array(self.fsv)

        if self.buy_count > self.sell_count:
            sell_prices.resize(buy_prices.size)
            ssv.resize(bbv.size)
            bs_len = self.buy_count
        else:
            buy_prices.resize(sell_prices.size)
            bbv.resize(ssv.size)
            bs_len = self.sell_count

        s_worth = sell_prices * ssv
        b_worth = buy_prices * bbv

        if self.filled_buy_count > self.filled_sell_count:
            filled_sell_prices.resize(filled_buy_prices.size)
            fsv.resize(fbv.size)
            fbs_len = self.filled_buy_count
        else:
            filled_buy_prices.resize(filled_sell_prices.size)
            fbv.resize(fsv.size)
            fbs_len = self.filled_sell_count

        fs_worth = filled_sell_prices * fsv
        fb_worth = filled_buy_prices * fbv

        fig = plt.figure(dpi=100, figsize=(12.8, 72))
        ax1 = fig.add_subplot(911)
        ax1.set_title("wp")
        ax1.plot(dt, wp, linewidth=0.5)
        ax2 = fig.add_subplot(912)
        ax2.set_title("action")
        ax2.bar(
            [
                "create",
                "fill",
                "wait",
                "buy",
                "sell",
                "fill_buy",
                "fill_sell"],
            [
                self.create_count,
                self.fill_count,
                self.wait_count,
                self.buy_volume,
                self.sell_volume,
                self.filled_buy_volume,
                self.filled_sell_volume],
            color=[
                'b',
                'y',
                'black',
                'r',
                'g',
                'r',
                'g'],
            label="fill ratio:{} bs_diff:{}".format(
                round(
                    self.fill_count /
                    self.create_count,
                    2),
                round(
                    self.filled_buy_volume -
                    self.filled_sell_volume,
                    2)))
        ax2.legend(fontsize=20)
        ax3 = fig.add_subplot(913)
        ax3.set_title(
            "buy sell price CD:{}".format(
                round(
                    np.mean(
                        self.cd_secs),
                    2)))
        ax3.plot(
            np.arange(bs_len),
            buy_prices,
            linewidth=0.8,
            color='green',
            label="buy mean:{}".format(
                round(
                    np.sum(buy_prices) /
                    self.buy_count,
                    4)))
        ax3.plot(
            np.arange(bs_len),
            sell_prices,
            linewidth=0.8,
            color='red',
            label="sell mean:{}".format(
                round(
                    np.sum(sell_prices) /
                    self.sell_count,
                    4)))
        ax3.legend(fontsize=20)
        ax4 = fig.add_subplot(914)
        ax4.set_title("filled buy sell price")
        ax4.plot(
            np.arange(fbs_len),
            filled_buy_prices,
            linewidth=0.8,
            color='green',
            label="fill buy mean:{}".format(
                round(
                    np.sum(filled_buy_prices) /
                    self.filled_buy_count,
                    4)))
        ax4.plot(
            np.arange(fbs_len),
            filled_sell_prices,
            linewidth=0.8,
            color='red',
            label="fill sell mean:{}".format(
                round(
                    np.sum(filled_sell_prices) /
                    self.filled_sell_count,
                    4)))
        ax4.legend(fontsize=20)
        ax5 = fig.add_subplot(915)
        ax5.set_title("profits")
        ax5.plot(np.arange(len(self.profits)), self.profits, linewidth=0.5,
                 label="total profit:{}".format(round(self.profits[-1], 4)))
        ax5.legend(fontsize=20)
        ax6 = fig.add_subplot(916)
        ax6.set_title("sharps")
        ax6.plot(np.arange(len(self.sharps)), self.sharps, linewidth=0.5,
                 label="final sharp:{}".format(round(self.sharps[-1], 4)))
        ax6.legend(fontsize=20)
        ax7 = fig.add_subplot(917)
        ax7.set_title("coin")
        ax7.scatter(np.arange(len(self.coin_state)), self.coin_state)
        ax8 = fig.add_subplot(918)
        ax8.set_title("buy sell info")
        ax8.bar(
            np.arange(bs_len),
            b_worth - s_worth,
            color='g',
            label="buy mean:{} sell mean:{}".format(
                round(
                    np.sum(b_worth) /
                    self.buy_count,
                    4),
                round(
                    np.sum(s_worth) /
                    self.sell_count,
                    4)))
        ax8.legend(fontsize=20)
        ax9 = fig.add_subplot(919)
        ax9.set_title("filled buy sell info")
        ax9.bar(
            np.arange(fbs_len),
            fb_worth -
            fs_worth,
            color='g',
            label="buy mean:{} sell mean:{}".format(
                round(
                    np.sum(fb_worth) /
                    self.filled_buy_count,
                    4),
                round(
                    np.sum(fs_worth) /
                    self.filled_sell_count,
                    4)))
        ax9.legend(fontsize=20)
        global FIG_PATH
        path = FIG_PATH + "\\" + self.model_name
        if not os.path.exists(path):
            os.mkdir(path)
        os.chdir(path)
        fig.savefig("{}.jpg".format(date))
        print("Save result fig to ", path)
        plt.close(fig)


def load_model(model_type: str, model_path: str) -> BaseAlgorithm:
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not exist!")

    if model_type.upper() == "PPO":
        return PPO.load(model_path)

    elif model_type.upper() == "SAC":
        return SAC.load(model_path)


def get_test_file_list(filepath=None) -> list:
    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError("Test file note exist! Check input file path!")

    file_list = os.listdir(filepath)
    for file in file_list:
        if not file[-4:] == ".csv":
            file_list.remove(file)

    return file_list


def main() -> None:
    global MODEL_PATH, MODEL_TYPE, DATA_PATH
    if len(sys.argv) > 3 or len(sys.argv) == 2:
        raise ValueError(
            "How to use: >>>python Model_Test.py model_type[PPO,SAC...] model_path")
    if len(sys.argv) == 3:
        MODEL_TYPE = sys.argv[1]
        MODEL_PATH = sys.argv[2]

    model = load_model(MODEL_TYPE, MODEL_PATH)
    model_name = MODEL_PATH.split('\\')[-1][:-4]
    env = TestEnv(model, model_name)
    file_list = get_test_file_list(DATA_PATH)

    for file in file_list:
        env.test(file)


if __name__ == '__main__':
    main()
