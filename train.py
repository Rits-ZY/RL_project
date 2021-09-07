import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
from stable_baselines3.common.base_class import BaseAlgorithm
from env.StockTradingEnv_2 import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy, window_func
import sys

LOG_PATH = "G:\\RL\\RL_project\\log\\"
DATA_PATH = "G:\\RL\\RL_project\\data\\shaped\\oneweek\\mydata20210402.csv"
MODEL_PATH = "G:\\RL\\RL_project\\model\\"
FIG_PATH = "G:\\RL\\RL_project\\fig\\"
MODEL_TYPE = "SAC"
DEFAULT_MAX_EPISODE = 4
DEFAULT_MODEL_NAME = "my_model"
MODEL_NAME = DEFAULT_MODEL_NAME
TRAIN_ROUND = 5


def load_model(
        model_type: str,
        train_env: VecMonitor) -> BaseAlgorithm:

    path = MODEL_PATH + "\\{}\\{}".format(MODEL_TYPE, MODEL_NAME)
    print("load model ", path)

    if not os.path.exists(path):
        raise FileNotFoundError("Model not exist!")

    if model_type.upper() == "PPO":
        return PPO.load(path, env=train_env, verbose=0)

    elif model_type.upper() == "SAC":
        return SAC.load(path, env=train_env, verbose=0)


def init_env():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values('ts')
    df_train = df.iloc[0:3600]
    print("Init train env done!")
    return DummyVecEnv([lambda: StockTradingEnv(df_train)])


def create_model(model_type: str, train_env: VecMonitor) -> BaseAlgorithm:
    if model_type.upper() == "PPO":
        return PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            n_steps=512,
            gae_lambda=0.98,
            n_epochs=10,
            batch_size=32,
            gamma=0.99)

    elif model_type.upper() == "SAC":
        return SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            batch_size=32,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1)


def save_fig(monitor_path: str, fig_path: str) -> None:
    data_frames = []
    data_frame = load_results(monitor_path)
    data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, "episodes") for data_frame in data_frames]
    max_x = max(xy[0][-1] for xy in xy_list)
    min_x = 0
    episodes_window = 100
    plt.figure("Train Rewards", figsize=(8, 2))
    for (_, (x, y)) in enumerate(xy_list):
        plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter
        # than window size.
        if x.shape[0] >= episodes_window:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, episodes_window, np.mean)
            plt.plot(x, y_mean)
    plt.xlim(min_x, max_x)
    plt.xlabel("episodes")
    plt.ylabel("Episode Rewards")
    plt.tight_layout()
    plt.savefig(fig_path)
    print("Save fig: ", fig_path)

def main(round: int) -> None:
    global MODEL_NAME, MODEL_TYPE, DEFAULT_MAX_EPISODE
    new_model_flag = False

    if len(sys.argv) > 4 or len(sys.argv) < 2 or len(sys.argv) == 3:
        raise ValueError(
            "How to use: >>>python train.py model_type[PPO,SAC...] model_path episode\n"
            "         or >>>python train.py model_type[PPO,SAC...]")

    if len(sys.argv) == 4:
        MODEL_TYPE = sys.argv[1]
        MODEL_NAME = sys.argv[2]
        DEFAULT_MAX_EPISODE = int(sys.argv[3])

    if len(sys.argv) == 2:
        MODEL_TYPE = sys.argv[1]
        new_model_flag = True

    train_env = init_env()
    log_path = ""
    if new_model_flag:
        model_name = DEFAULT_MODEL_NAME
        log_path = LOG_PATH + "\\{}\\{}".format(MODEL_TYPE, model_name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        log_path_2 = log_path + "\\{}_{}".format(round, Monitor.EXT)
        if os.path.exists(log_path_2):
            log_path = log_path_2
        train_env = VecMonitor(train_env, log_path)
        model = create_model(MODEL_TYPE, train_env)

    else:
        model_name = MODEL_NAME[:-4]
        log_path = LOG_PATH + \
            "\\{}\\{}\\{}_{}".format(MODEL_TYPE, model_name, round, Monitor.EXT)
        if not os.path.exists(log_path):
            print("Create new file:", log_path)
        train_env = VecMonitor(train_env, log_path)
        model = load_model(MODEL_TYPE, train_env)

    my_callback = StopTrainingOnMaxEpisodes(DEFAULT_MAX_EPISODE, verbose=0)
    model.learn(total_timesteps=20000000, callback=my_callback)
    save_path = MODEL_PATH + \
        "\\{}\\{}_{}.zip".format(MODEL_TYPE, round, model_name)
    model.save(save_path)
    print("Save model: ", save_path)

    fig_path = FIG_PATH + \
        "\\{}\\train\\{}_{}.jpg".format(MODEL_TYPE, round, model_name)
    save_fig(os.path.dirname(log_path), fig_path)


if __name__ == '__main__':
    for i in range(TRAIN_ROUND):
        main(i)
