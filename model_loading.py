import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import LSTM, KF, load_test, getMonteCarlo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 參數設置
hidden_size = 128
start_size = 9999
validation_size = 5000
data_set_size = start_size + validation_size

# 準備輸入數據
# 讀取檔案
# path1 = ['E:/cmake_mouse_boundary_v9_1/build/IPS750_G50_F_motion.txt'] #馬達資料.txt路徑
# path2 = ['E:/cmake_mouse_boundary_v9_1/build/IPS750_G50_F_mouse.txt']  #滑鼠資料.txt路徑
# x_kf_update_data, P_kf_update_data, K_update_data, k_y_update_data, KCP_data, H, Pos, PosCmd, VelCmd, AccCmd, PosCmd_AddNoise, VelCmd_AddNoise, AccCmd_AddNoise = KF.KF_Process(path1, path2)
# x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, x_true_data, x_true_data_noise, P_k_data, prediction_errors_data = KF_training_data.KF_Process(data_set_size)
# z = PosCmd
# x_true = cp.array([PosCmd, 
#                    VelCmd,
#                    AccCmd])

x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, x_k_predict_data, P_data, P_k_update_data, KCP_data, P_input_data_all = load_test.load_data('x_data_all.txt', 'P_data_all.txt')
# x_input_data_all = np.loadtxt('x_input_data_all_normalized.txt', delimiter=' ')
# P_input_data_all = np.loadtxt('P_input_data_all_normalized.txt', delimiter=' ')

# --------x_model loading --------#
# 設置模型參數
input_size = 6  # 根據需要的特徵數
output_size = 2    # 假設輸出有兩個維度

# 加載模型
x_lstm_model_loaded = LSTM.LSTM_KF(input_size, hidden_size, output_size)  # 創建模型實例
x_lstm_model_loaded.load_state_dict(torch.load('x_lstm_kf_model_t1.pth', weights_only=True))  # 加載權重
x_lstm_model_loaded.eval()  # 將模型設置為評估模式
x_lstm_model_loaded = x_lstm_model_loaded.to(device)

x_lstm_output_data = []

for k in range(start_size, start_size + validation_size):
    print("k =", k)
    x_tel = cp.array(x_true) - cp.array(x_k_update_data)
    x_input_data = x_input_data_all[k]
    x_input_data = torch.tensor(cp.hstack(x_input_data), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x_input_tensor = x_input_data.clone().detach().to(device)

    # 使用模型進行推斷
    with torch.no_grad():  # 禁用梯度計算以提高推斷效率
        x_lstm_output = x_lstm_model_loaded(x_input_tensor)  # 獲取模型的輸出
    x_lstm_output_data.append(x_lstm_output.detach().cpu().numpy().flatten())
    print("x LSTM Output:", x_lstm_output[:, :2])  # 輸出結果

# 解歸一化
# x_lstm_output_data_pd = pd.DataFrame(x_lstm_output_data)
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler.fit(x_lstm_output_data)
# x_lstm_output_data_denormalized = pd.DataFrame(scaler.inverse_transform(x_lstm_output_data), columns=x_lstm_output_data_pd.columns)
# print("x =",cp.reshape(x_lstm_output_data_denormalized.iloc[-1, :2].to_numpy(), (2, 1)))
print("x =",cp.reshape(cp.array(x_lstm_output_data)[-1, :2], (2, 1)))

# --------P_model loading --------#
# 設置模型參數
input_size = 8  # 根據需要的特徵數
output_size = 4   # 假設輸出有兩個維度

# 加載模型
P_lstm_model_loaded = LSTM.LSTM_KF(input_size, hidden_size, output_size)  # 創建模型實例
P_lstm_model_loaded.load_state_dict(torch.load('P_lstm_kf_model_t1.pth', weights_only=True))  # 加載權重
P_lstm_model_loaded.eval()  # 將模型設置為評估模式
P_lstm_model_loaded = P_lstm_model_loaded.to(device)

P_lstm_output_data = []

for k in range(start_size, start_size + validation_size):
    P_input_data = torch.tensor(cp.hstack(P_input_data_all[k]), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    P_input_tensor = P_input_data.clone().detach().to(device)

    # 使用模型進行推斷
    with torch.no_grad():  # 禁用梯度計算以提高推斷效率
        P_lstm_output = P_lstm_model_loaded(P_input_tensor)  # 獲取模型的輸出
    P_lstm_output_data.append(P_lstm_output.detach().cpu().numpy().flatten())

# 將數據解歸一化
# P_lstm_output_data_pd = pd.DataFrame(P_lstm_output_data)
# scaler.fit(P_lstm_output_data)
# P_lstm_output_data_denormalized = pd.DataFrame(scaler.inverse_transform(P_lstm_output_data), columns=P_lstm_output_data_pd.columns)
# print("P =",cp.reshape(P_lstm_output_data_denormalized.iloc[-1, :4].to_numpy(), (2, 2)))
print("P =",cp.reshape(cp.array(P_lstm_output_data)[-1, :4], (2, 2)))

# 先将 cupy 数组转换为 numpy 数组
x_lstm_output_data_np = x_lstm_output_data.get()
P_lstm_output_data_np = P_lstm_output_data.get()
# 使用 numpy.savetxt 将其保存到 txt 文件中
np.savetxt('x_lstm_output_data.txt', x_lstm_output_data_np, delimiter=' ')
np.savetxt('P_lstm_output_data.txt', P_lstm_output_data_np, delimiter=' ')

# 估測狀態匯出
plt.figure()
# plt.plot(x_true[start_size:start_size + validation_size, 0], label='True_x1', color='black', linewidth=3)
# plt.plot(x_true[start_size:start_size + validation_size, 1], label='True_x2', color='blue', linewidth=3)
plt.plot(x_true_noise[start_size:start_size + validation_size, 0], label='True_x1_add_noise', color='black', linewidth=3)
plt.plot(x_true_noise[start_size:start_size + validation_size, 1], label='True_x2_add_noise', color='blue', linewidth=3)
plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0].get(), label='LKF_x1', color='orange', linewidth=2)
plt.plot(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1].get(), label='LKF_x2', color='cyan', linewidth=2)
plt.plot(cp.array(x_lstm_output_data)[:, 0].get(), label='DKF_x1', color='purple', linewidth=1)
plt.plot(cp.array(x_lstm_output_data)[:, 1].get(), label='DKF_x2', color='red', linewidth=1)
plt.xlabel('data')
plt.ylabel('value')
plt.legend()
plt.title('estimate vs true :x1 x2')

# 估測狀態誤差匯出
plt.figure()
a = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
plt.plot(a.get(), label='LKF_x1', color='black', linewidth=2)
b = cp.abs(cp.array(x_k_update_data)[start_size:start_size + validation_size, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
plt.plot(b.get(), label='LKF_x2', color='green', linewidth=2)
c = cp.abs(cp.array(x_lstm_output_data)[:, 0] - cp.array(x_true)[start_size:start_size + validation_size, 0])
plt.plot(c.get(), label='DKF_x1', color='blue', linewidth=1)
d = cp.abs(cp.array(x_lstm_output_data)[:, 1] - cp.array(x_true)[start_size:start_size + validation_size, 1])
plt.plot(d.get(), label='DKF_x2', color='red', linewidth=1)
plt.xlabel('data')
plt.ylabel('estimate value')
plt.legend()
plt.title('estimate pos vel acc')

plt.show()