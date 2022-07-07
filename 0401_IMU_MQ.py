import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import util
import SSA
from skimage.transform import resize


plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] ='Malgun Gothic'

data_path = "./data/2022_04_01_motion_imu/"
fn_list = os.listdir(data_path)
# user_list = ["인", "데이비드", "제이슨", "로제", "마이어", "찰스", "지나", "제임스", "미쉘"]
user_list = ["인", "지나"]
leg_score = {"인": 31.46, "데이비드": 36.62, "제이슨": 17.51, "로제": 22.75, "마이어": 38.76, "찰스": 41.19, "지나": 20.04,
                     "제임스": 31.66, "미쉘": 13.35}

dic_list = {}

for fn in fn_list:
    user_name = fn.split("_")[0]

    if user_name not in user_list:
        continue

    sensor_name = fn.split("_")[-1].split(".")[0]
    condition = fn.split("_")[-2]
    if sensor_name != "M1" or condition != "왼발디딤":
        continue

    if user_name not in dic_list:
        dic_list[user_name] = {}
        dic_list[user_name]["y"] = leg_score[user_name]

    df = pd.read_csv(f"{data_path}{fn}", sep=",", names=["ax", "ay", "az", "gx", "gy", "gz"])
    np_data = df.to_numpy()

    # ssa_list = []
    # stft_list = []
    # for i in range(6):
    #     ssa = SSA.SSA(np_data[:, i], L=20)
    #     ssa_list.append(ssa.reconstruct([0]).to_numpy())
    #     stft = resize(util.stft(ssa.reconstruct([0]), 64), (32, 128), mode='constant')
    #     stft2 = resize(util.stft(np_data[:, i], 64), (32, 128), mode='constant')
    #
    #     ax = plt.imshow(stft)
    #     plt.title(fn)
    #     plt.show()
    #     stft_list.append(stft)
    #
    #     ax = plt.imshow(stft2)
    #     plt.title(fn)
    #     plt.show()
    #     stft_list.append(stft2)
    #
    # np_ssa_list = np.array(ssa_list)
    # np_stft_list = np.array(stft_list)
    #
    # if "stft" in dic_list[user_name]:
    #     dic_list[user_name]["stft"].append(np_stft_list)
    # else:
    #     dic_list[user_name]["stft"] = [np_stft_list]

    for i in [3]:
        ssa = SSA.SSA(np_data[:, i], L=20)
        recur_plot = util.recurrence_plot(ssa.reconstruct([0]).to_numpy().reshape(-1, 1), 50000)
        ax = plt.imshow(recur_plot)
        plt.title(f"{fn.split('/')[-1]}")
        plt.clim(0, 10000)
        plt.colorbar(ax)
        plt.show()

