import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import LSTM, dataset_arrange
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time

# 參數設置
hidden_size = 64
start_size = 9999
validation_size = 5000
data_set_size = start_size + validation_size

# 準備輸入數據
# 模擬資料
path1 = './sim_dataset/x_data_all_15000_0.001.txt'
path2 = './sim_dataset/P_data_all_15000_0.001.txt'
x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, x_k_predict_data, P_data, P_k_update_data, KCP_data, P_input_data_all = dataset_arrange.loadSimData_less_feature(path1, path2)
# x_input_data_all = np.loadtxt('x_input_data_all_normalized.txt', delimiter=' ')
# P_input_data_all = np.loadtxt('P_input_data_all_normalized.txt', delimiter=' ')

# --------x_model loading --------#
# 設置模型參數
input_size = 2  # 根據需要的特徵數
output_size = 2    # 假設輸出有兩個維度

# 加載模型
x_lstm_model_loaded = LSTM.LSTM_KF(input_size, hidden_size, output_size)  # 創建模型實例
x_lstm_model_loaded.load_state_dict(torch.load('sim/model/x_model_0.001_layer1_fea1_2.pth', weights_only=True))  # 加載權重
x_lstm_model_loaded.eval()  # 將模型設置為評估模式
x_lstm_model_loaded = x_lstm_model_loaded.to(device)

# --------P_model loading --------#
# 設置模型參數
input_size = 8  # 根據需要的特徵數
output_size = 4   # 假設輸出有兩個維度

# 加載模型
P_lstm_model_loaded = LSTM.LSTM_KF(input_size, hidden_size, output_size)  # 創建模型實例
P_lstm_model_loaded.load_state_dict(torch.load('sim/model/P_model_0.001_layer1_fea1_2.pth', weights_only=True))  # 加載權重
P_lstm_model_loaded.eval()  # 將模型設置為評估模式
P_lstm_model_loaded = P_lstm_model_loaded.to(device)

start_time = time.time()
x_lstm_output_data = []
for k in range(start_size, start_size + validation_size):
    # print("k =", k)
    # x_tel = cp.array(x_true) - cp.array(x_k_update_data)
    x_input_data = x_input_data_all[k]
    x_input_data = torch.tensor(cp.hstack(x_input_data), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x_input_tensor = x_input_data.clone().detach().to(device)

    # 使用模型進行推斷
    with torch.no_grad():  # 禁用梯度計算以提高推斷效率
        x_lstm_output = x_lstm_model_loaded(x_input_tensor)  # 獲取模型的輸出
    x_lstm_output_data.append(x_lstm_output.detach().cpu().numpy().flatten())
    # print("x LSTM Output:", x_lstm_output[:, :3].cpu().numpy())  # 輸出結果

P_lstm_output_data = []
for k in range(start_size, start_size + validation_size):
    P_input_data = torch.tensor(cp.hstack(P_input_data_all[k]), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    P_input_tensor = P_input_data.clone().detach().to(device)

    # 使用模型進行推斷
    with torch.no_grad():  # 禁用梯度計算以提高推斷效率
        P_lstm_output = P_lstm_model_loaded(P_input_tensor)  # 獲取模型的輸出
    P_lstm_output_data.append(P_lstm_output.detach().cpu().numpy().flatten())
end_time = time.time()
# 先将 cupy 数组转换为 numpy 数组
# x_lstm_output_data_np = x_lstm_output_data.get()
# P_lstm_output_data_np = P_lstm_output_data.get()
x_lstm_output_data_np = x_lstm_output_data
P_lstm_output_data_np = P_lstm_output_data
# 使用 numpy.savetxt 将其保存到 txt 文件中
np.savetxt('./result/x_lstm_output_data_sim.txt', x_lstm_output_data_np, delimiter=' ')
np.savetxt('./result/P_lstm_output_data_sim.txt', P_lstm_output_data_np, delimiter=' ')

print("一個特徵DKF運行時間 :", end_time - start_time)
print("一個特徵DKF平均一筆運行時間 :", (end_time - start_time) / validation_size)
# 估測狀態匯出
plt.figure()
# x_true_noise = x_true
# plt.plot(x_true[start_size:start_size + validation_size, 0], label='True_x1', color='black', linewidth=3)
# plt.plot(x_true[start_size:start_size + validation_size, 1], label='True_x2', color='blue', linewidth=3)
plt.plot(x_true_noise[start_size:start_size + validation_size, 0], label='True_x1_add_noise', color='black', linewidth=3)
plt.plot(x_true_noise[start_size:start_size + validation_size, 1], label='True_x2_add_noise', color='blue', linewidth=3)
# print("x_k_update_data =", x_k_update_data)
plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0].get(), label='LKF_x1', color='orange', linewidth=2)
plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1].get(), label='LKF_x2', color='cyan', linewidth=2)
# plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2].get(), label='LKF_x3', color='pink', linewidth=2)
plt.plot(cp.array(x_lstm_output_data)[3:, 0].get(), label='DKF_x1', color='purple', linewidth=1)
plt.plot(cp.array(x_lstm_output_data)[3:, 1].get(), label='DKF_x2', color='red', linewidth=1)
# plt.plot(cp.array(x_lstm_output_data)[:, 2].get(), label='DKF_x3', color='green', linewidth=1)
plt.xlabel('data')
plt.ylabel('value')
plt.legend()
plt.title('estimate vs true :x1 x2')

# 估測狀態誤差匯出
plt.figure()
# x_k_update_data = cp.array(x_k_update_data).reshape(-1, 1)
# print("x_k_update_data =", x_k_update_data)
# print("x_true =", x_true)
a = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
plt.plot(a.get(), label='LKF_x1', color='black', linewidth=2)
b = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
plt.plot(b.get(), label='LKF_x2', color='green', linewidth=2)
# e = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2] - cp.array(x_true)[start_size:start_size + validation_size, 2])
# plt.plot(e.get(), label='LKF_x2', color='purple', linewidth=2)
c = cp.abs(cp.array(x_lstm_output_data)[:, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
plt.plot(c.get(), label='DKF_x1', color='blue', linewidth=1)
d = cp.abs(cp.array(x_lstm_output_data)[:, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
plt.plot(d.get(), label='DKF_x2', color='red', linewidth=1)
# f = cp.abs(cp.array(x_lstm_output_data)[:, 2] - cp.array(x_true)[start_size:start_size + validation_size, 2])
# plt.plot(f.get(), label='DKF_x2', color='red', linewidth=1)
plt.xlabel('data')
plt.ylabel('estimate value')
plt.legend()
plt.title('estimate pos vel acc')

print("---------------------------------------")
print("LKF_x1 mean error :", np.mean(a))
print("LKF_x2 mean error :", np.mean(b))
print("DKF_x1 mean error :", np.mean(c))
print("DKF_x2 mean error :", np.mean(d))
print("---------------------------------------")

plt.show()