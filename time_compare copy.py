import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import LSTM, dataset_arrange
from sim.training_data_sim import KalmanFilter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 參數設置
hidden_size = 64
start_size = 9999
validation_size = 5000
data_set_size = start_size + validation_size

A_data = []
# 狀態轉移矩陣
k = 15000
for k in range(k):
    # print(k)
    # k = k 
    A = cp.array([[0.7 + 0.05 * cp.sin(0.001*k), 0.06 + 0.04 * cp.arctan(1 / (k + 1))],
                [0.10 - 0.1 * cp.sin(0.001*k), -0.2 + 0.2 * cp.sin(0.001*k)]])
    A_data.append(A)
B = cp.array([[0],
              [0]])
H = cp.array([[1.8, 2]]) # 1/cpi 
# 過程噪聲 
Q = cp.array([[0.8, 0],
            [0, 1.2]]) 
# 測量噪聲
R = 0.95 #與誤差有關 -> 影響平滑度
# 誤差協方差矩陣
P = cp.array([[1e-4, 1e-4],
            [1e-4, 1e-4]])
x = cp.array([[0.2],
            [0.5]]) 
# 控制輸入
u = cp.zeros((1, 1))

# 準備輸入數據
# 模擬資料
path1 = './sim_dataset/x_data_all_15000_0.05.txt'
path2 = './sim_dataset/P_data_all_15000_0.05.txt'
x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, x_k_predict_data, P_data, P_k_update_data, KCP_data, P_input_data_all = dataset_arrange.loadSimData(path1, path2)

# --------x_model loading --------#
# 設置模型參數
input_size = 6  # 根據需要的特徵數
output_size = 2    # 假設輸出有兩個維度

# 加載模型
x_lstm_model_loaded = LSTM.LSTM_KF(input_size, hidden_size, output_size)  # 創建模型實例
x_lstm_model_loaded.load_state_dict(torch.load('sim/model/x_lstm_kf_model_layer1_2.pth', weights_only=True))  # 加載權重
x_lstm_model_loaded.eval()  # 將模型設置為評估模式
x_lstm_model_loaded = x_lstm_model_loaded.to(device)
# --------P_model loading --------#
# 設置模型參數
input_size = 8  # 根據需要的特徵數
output_size = 4   # 假設輸出有兩個維度

# 加載模型
P_lstm_model_loaded = LSTM.LSTM_KF(input_size, hidden_size, output_size)  # 創建模型實例
P_lstm_model_loaded.load_state_dict(torch.load('sim/model/P_lstm_kf_model_layer1_2.pth', weights_only=True))  # 加載權重
P_lstm_model_loaded.eval()  # 將模型設置為評估模式
P_lstm_model_loaded = P_lstm_model_loaded.to(device)

# DNNKF
start_time1 = time.time()
x_lstm_output_data = []
P_lstm_output_data = []
x_true_data = x_true
x_true_data_noise = x_true_noise

for k in range(start_size, start_size + validation_size):
    # x_k = x_true_data[k].reshape(2, 1) + B @ u # x_k為預測的x
    x_k = A_data[k] @ x + B @ u # x_k為預測的x
    P_k = A_data[k] @ P @ A_data[k].T + Q # P_k為預測的x
    # 卡爾曼增益
    K = P_k @ H.T @ cp.linalg.inv(H @ P_k @ H.T + R)
    # 狀態更新
    z  = H @ cp.array(x_true_noise)[k]
    y_tel = (z - H @ x_k) # z為實際觀測到的
    k_y = K @ y_tel
    # k_y_data.append(k_y)
    x_k_1 = x_k + k_y # x_k_1為更新後的x
    x = x_k_1 # x更新後跌代
    x_k_update = x
    x_tel2 = cp.array(x_true_data[k].reshape(2, 1)) - x_k_update
    x_k_update_data.append(x.flatten())
    # x_k_predict_data.append(x_k.flatten())
    # 誤差協方差更新
    KCP = K @ H @ P_k
    # KCP_data.append(KCP.flatten())
    P_k_1 = (cp.eye((K @ H).shape[0]) - K @ H) @ P_k # P_k_1為更新後的x
    P = P_k_1

# --------x_model loading --------#
    # print("k =", k)
    x_input_data = np.concatenate((x_k_update, k_y, x_tel2), axis=1)
    x_input_data = torch.tensor(cp.hstack(x_input_data), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x_input_tensor = x_input_data.clone().detach().to(device)

    # 使用模型進行推斷
    with torch.no_grad():  # 禁用梯度計算以提高推斷效率
        x_lstm_output = x_lstm_model_loaded(x_input_tensor)  # 獲取模型的輸出
    x_lstm_output_data.append(x_lstm_output.detach().cpu().numpy().flatten())
    x_lstm_output
    # print("x LSTM Output:", x_lstm_output[:, :3])  # 輸出結果
# end_time1 = time.time()

# --------P_model loading --------#
# P_lstm_output_data = []
# for k in range(start_size, start_size + validation_size):
    P_input_data = np.concatenate((P, KCP), axis=1)
    P_input_data = torch.tensor(cp.hstack(P_input_data), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    P_input_tensor = P_input_data.clone().detach().to(device)

    # 使用模型進行推斷
    with torch.no_grad():  # 禁用梯度計算以提高推斷效率
        P_lstm_output = P_lstm_model_loaded(P_input_tensor)  # 獲取模型的輸出
    P_lstm_output_data.append(P_lstm_output.detach().cpu().numpy().flatten())
end_time3 = time.time()

# KF
start_time2 = time.time()
KF = KalmanFilter()
x_k_update_data_KF_data = []
for k in range(start_size, start_size + validation_size):
    # print(k+1)
    # print("12312313")
    # 卡爾曼濾波器更新步驟
    # x_true_data = cp.asarray(x_true)
    x_k_update_data_KF, x_k_predict_data_KF, P_k_update_data_KF, P_k_predict_data_KF, k_y_data_KF, KCP_data_KF, z_data_KF, P_k_data_KF = KF.KF_time(k, A_data, x_true, x_true_noise)
    # print("x_k_update_data_KF =", x_k_update_data_KF)
    # x_k_update_data_KF_data.append(x_k_update_data_KF.flatten())
    # x_k_update_data_KF = cp.asarray(x_k_update_data_KF)
    # x_k_update_data_KF_data = cp.vstack([x_k_update_data_KF_data, x_k_update_data_KF.flatten()])
    # print(k+1)
end_time2 = time.time()

# 計算時長
# DNNKF_time = end_time1 - start_time1
LKF_time = end_time2 - start_time2
DNNKF_total_time = end_time3 - start_time1
# print("DNNKF僅估計x所花時間:", DNNKF_time)
# print("DNNKF僅估計x平均一筆所花時間:", DNNKF_time/validation_size)
print("LKF_time所花時間:", LKF_time)
print("LKF_time平均一筆所花時間:", LKF_time/validation_size)
print("DNNKF估計x和P平均一筆所花時間:", DNNKF_total_time/validation_size)

# 先将 cupy 数组转换为 numpy 数组
# x_lstm_output_data_np = x_lstm_output_data.get()
# P_lstm_output_data_np = P_lstm_output_data.get()
# x_lstm_output_data_np = x_lstm_output_data
# P_lstm_output_data_np = P_lstm_output_data
# 使用 numpy.savetxt 将其保存到 txt 文件中
# np.savetxt('./result/x_lstm_output_data_sim.txt', x_lstm_output_data_np, delimiter=' ')
# np.savetxt('./result/P_lstm_output_data_sim.txt', P_lstm_output_data_np, delimiter=' ')

# 估測狀態匯出
plt.figure()
# x_true_noise = x_true
plt.plot(x_true_noise[start_size:start_size + validation_size, 0], label='True_x1_add_noise', color='black', linewidth=3)
plt.plot(x_true_noise[start_size:start_size + validation_size, 1], label='True_x2_add_noise', color='blue', linewidth=3)
plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0].get(), label='LKF_x1', color='orange', linewidth=2)
plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1].get(), label='LKF_x2', color='cyan', linewidth=2)
plt.plot(cp.array(x_lstm_output_data)[:, 0].get(), label='DKF_x1', color='purple', linewidth=1)
plt.plot(cp.array(x_lstm_output_data)[:, 1].get(), label='DKF_x2', color='red', linewidth=1)
plt.xlabel('data')
plt.ylabel('value')
plt.legend()
plt.title('DNNKF estimate vs true :x1 x2')

# 估測狀態誤差匯出
plt.figure()
# print("x_k_update_data =", x_k_update_data)
# print("x_true =", x_true)
a = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
plt.plot(a.get(), label='LKF_x1', color='orange', linewidth=2)
b = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
plt.plot(b.get(), label='LKF_x2', color='cyan', linewidth=2)
c = cp.abs(cp.array(x_lstm_output_data)[:, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
plt.plot(c.get(), label='DKF_x1', color='purple', linewidth=1)
d = cp.abs(cp.array(x_lstm_output_data)[:, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
plt.plot(d.get(), label='DKF_x2', color='red', linewidth=1)
plt.xlabel('data')
plt.ylabel('estimate value')
plt.legend()
plt.title('estimate pos vel acc')

# KF
plt.figure()
x_k_update_data_KF = cp.array(x_k_update_data_KF)
# x_true_noise = x_true
plt.plot(x_true_noise[start_size:start_size + validation_size, 0], label='True_x1_add_noise', color='black', linewidth=3)
plt.plot(x_true_noise[start_size:start_size + validation_size, 1], label='True_x2_add_noise', color='blue', linewidth=3)
# plt.plot(cp.array(x_k_update_data_KF)[start_size:start_size + validation_size, 0].get(), label='LKF_x1', color='orange', linewidth=2)
# plt.plot(cp.array(x_k_update_data_KF)[start_size:start_size + validation_size, 1].get(), label='LKF_x2', color='cyan', linewidth=2)
plt.plot(cp.array(x_k_update_data_KF)[:,0].get(), label='LKF_x1', color='orange', linewidth=2)
plt.plot(cp.array(x_k_update_data_KF)[:,1].get(), label='LKF_x2', color='cyan', linewidth=2)
# print("x_k_update_data_KF_data =", x_k_update_data_KF_data)
plt.xlabel('data')
plt.ylabel('value')
plt.legend()
plt.title('LKF estimate vs true :x1 x2')

plt.show()