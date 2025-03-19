import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import zero_phase_filter

def CFD(data):
    # data = []
    SamplingTime = 0.001

    est1 = data #濾波後的數值
    # for i in range(1, len(Pos)):
    #     if Pos[i] < 0:
    #         est1[i] = -est1[i]
    # 计算一阶导数
    est2 = np.concatenate(([0], (est1[2:] - est1[:-2]) / (2 * SamplingTime), [0])) #把濾波後的數值待入中央差分法得到速度
    # 计算二阶导数
    est3 = np.concatenate(([0], (est1[2:] - 2 * est1[1:-1] + est1[:-2]) / (SamplingTime ** 2), [0])) #把濾波後的數值待入中央差分法得到加速度
    
    return est1, est2, est3

def CFD_2(data):
    # data = []
    global SamplingTime
    SamplingTime = 0.001

    est1 = data #濾波後的數值
    # for i in range(1, len(Pos)):
    #     if Pos[i] < 0:
    #         est1[i] = -est1[i]

    # 计算一阶导数
    est2 = np.concatenate(([0], (est1[2:] - est1[:-2]) / (2 * SamplingTime), [0])) #把濾波後的數值待入中央差分法得到速度
    # 计算二阶导数
    # est3 = np.concatenate(([0], (est1[2:] - 2 * est1[1:-1] + est1[:-2]) / (SamplingTime ** 2), [0])) #把濾波後的數值待入中央差分法得到加速度
    
    return est2