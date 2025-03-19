import cupy as cp
# import numpy as cp
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import LSTM, dataset_arrange
import setLSTMConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 馬達實際資料
path1 = 'sim_data/dataset/x_data_all_15000_Q2.txt'
path2 = 'sim_data/dataset/P_data_all_15000_Q2.txt'
path3 = 'sim_data/dataset/raw_data_all_15000_Q2.txt'
x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, raw_data_all  = dataset_arrange.loadSimData(path1, path2, path3)

# 參數設置
setConfig = setLSTMConfig.LSTMConfig()
x_input_size, x_output_size, hidden_size, num_layers, dropout, P_input_size, P_output_size = setConfig.getLSTMConfig()

# 加載模型
x_lstm_model_loaded = LSTM.LSTM_KF(x_input_size, hidden_size, x_output_size, num_layers, dropout)  # 創建模型實例
x_lstm_model_loaded.load_state_dict(torch.load('sim_data/model/x_model.pth', weights_only=True))  # 加載權重
x_lstm_model_loaded.eval()  # 將模型設置為評估模式
x_lstm_model_loaded = x_lstm_model_loaded.to(device)
# hidden_size = 128
start_size = 10000
validation_size = 5000 # diff: 14683 # same: 23000
data_set_size = start_size + validation_size

P_true = cp.array([[1e-7, 1e-7, 1e-7],
                   [1e-7, 1e-7, 1e-7],
                   [1e-7, 1e-7, 1e-7]])

# # 加載模型
# P_lstm_model_loaded = LSTM.LSTM_KF(input_size, hidden_size, output_size)  # 創建模型實例
# P_lstm_model_loaded.load_state_dict(torch.load('sim_data/model/P_model.pth', weights_only=True))  # 加載權重
# P_lstm_model_loaded.eval()  # 將模型設置為評估模式
# P_lstm_model_loaded = P_lstm_model_loaded.to(device)

# --------x_model loading --------#
x_lstm_output_data = []
for k in range(start_size, start_size + validation_size):
    print("k =", k)
    # x_tel = cp.array(x_true) - cp.array(x_k_update_data)
    x_input_data = x_input_data_all[k]
    x_input_data = torch.tensor(cp.hstack(x_input_data), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x_input_tensor = x_input_data.clone().detach().to(device)

    # 使用模型進行推斷
    with torch.no_grad():  # 禁用梯度計算以提高推斷效率
        x_lstm_output = x_lstm_model_loaded(x_input_tensor)  # 獲取模型的輸出
    x_lstm_output_data.append(x_lstm_output.detach().cpu().numpy().flatten())
    print("x LSTM Output:", x_lstm_output[:, :3])  # 輸出結果
print('-----------------')
print("x =",cp.reshape(cp.array(x_lstm_output_data)[-1, :3], (3, 1)))

# --------P_model loading --------#
# P_lstm_output_data = []
# for k in range(start_size, start_size + validation_size):
#     P_input_data = torch.tensor(cp.hstack(P_input_data_all[k]), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
#     P_input_tensor = P_input_data.clone().detach().to(device)

#     # 使用模型進行推斷
#     with torch.no_grad():  # 禁用梯度計算以提高推斷效率
#         P_lstm_output = P_lstm_model_loaded(P_input_tensor)  # 獲取模型的輸出
#     P_lstm_output_data.append(P_lstm_output.detach().cpu().numpy().flatten())
# print("P =",cp.reshape(cp.array(P_lstm_output_data)[-1, :9], (3, 3)))

# 先将 cupy 数组转换为 numpy 数组
# x_lstm_output_data_np = x_lstm_output_data.get()
# P_lstm_output_data_np = P_lstm_output_data.get()
x_lstm_output_data_np = x_lstm_output_data
# P_lstm_output_data_np = P_lstm_output_data
# 使用 numpy.savetxt 将其保存到 txt 文件中
np.savetxt('sim_data/result/x_lstm_output_sim.txt', x_lstm_output_data_np, delimiter=' ')
# np.savetxt('./result/P_lstm_output_data_motor.txt', P_lstm_output_data_np, delimiter=' ')

# print("Plotting ...")
# 估測狀態匯出
plt.figure()
x_true_noise = x_true
# plt.plot(x_true_noise[start_size:start_size + validation_size], label='True_x1_add_noise', color='black', linewidth=3)
plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0].get(), label='LKF_pos', color='orange', linewidth=2)
plt.plot(cp.array(x_lstm_output_data)[:, 0].get(), label='DKF_pos', color='purple', linewidth=1)
plt.plot(raw_data_all[start_size:start_size + validation_size, 0], label='true_pos', color='black', linewidth=1)
plt.xlabel('data')
plt.ylabel('value')
plt.legend()
plt.title('Pos of estimate vs true')

plt.figure()
x_true_noise = x_true
plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1].get(), label='LKF_vel', color='cyan', linewidth=2)
plt.plot(cp.array(x_lstm_output_data)[:, 1].get(), label='DKF_vel', color='red', linewidth=1)
plt.plot(raw_data_all[start_size:start_size + validation_size, 1], label='true_vel', color='black', linewidth=1)
plt.xlabel('data')
plt.ylabel('value')
plt.legend()
plt.title('Vel of estimate vs true')

plt.figure()
x_true_noise = x_true
plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2].get(), label='LKF_acc', color='pink', linewidth=2)
plt.plot(cp.array(x_lstm_output_data)[:, 2].get(), label='DKF_acc', color='green', linewidth=1)
plt.plot(raw_data_all[start_size:start_size + validation_size, 2], label='true_acc', color='black', linewidth=1)
plt.xlabel('data')
plt.ylabel('value')
plt.legend()
plt.title('Acc of estimate vs true')

# 估測狀態誤差匯出
x_k_update_data = cp.array(x_k_update_data).reshape(-1, 1)  # reshape to 2D
x_true = cp.array(x_true).reshape(-1, 1)  # reshape to 2D
plt.figure()
a = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
plt.plot(a.get(), label='LKF_x1_err', color='black', linewidth=2)
# b = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
# plt.plot(b.get(), label='LKF_x2', color='green', linewidth=2)
# e = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 2] - cp.array(x_true)[start_size:start_size + validation_size, 2])
# plt.plot(e.get(), label='LKF_x3', color='purple', linewidth=2)
c = cp.abs(cp.array(x_lstm_output_data)[:, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
plt.plot(c.get(), label='DKF_x1_err', color='blue', linewidth=1)
# d = cp.abs(cp.array(x_lstm_output_data)[:, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
# plt.plot(d.get(), label='DKF_x2', color='red', linewidth=1)
# f = cp.abs(cp.array(x_lstm_output_data)[:, 2] - cp.array(x_true)[start_size:start_size + validation_size, 2])
# plt.plot(f.get(), label='DKF_x3', color='red', linewidth=1)
plt.xlabel('data')
plt.ylabel('estimate value')
plt.legend()
plt.title('pos error between true_pos')
# print("Finish Plotting ...")

plt.show()