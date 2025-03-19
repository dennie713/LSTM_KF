import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import LSTM, dataset_arrange
import setLSTMConfig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 訓練參數設置
epoch = 300
traning_size = 10000   # diff: 14683 ；diff_2: 29366 # same: 23000
batch_size = 100
data_set_size = traning_size

setConfig = setLSTMConfig.LSTMConfig()
x_input_size, x_output_size, hidden_size, num_layers, dropout, P_input_size, P_output_size = setConfig.getLSTMConfig()
# x初始化LSTM模型
x_lstm_model = LSTM.LSTM_KF(x_input_size, hidden_size, x_output_size, num_layers, dropout)
x_lstm_model = x_lstm_model.to(device)
x_optimizer = torch.optim.Adam(x_lstm_model.parameters(), lr=0.001)
x_loss_fn = nn.MSELoss()

# P初始化LSTM模型
P_lstm_model = LSTM.LSTM_KF(P_input_size, hidden_size, P_output_size, num_layers, dropout)
P_lstm_model = P_lstm_model.to(device)
P_optimizer = torch.optim.Adam(P_lstm_model.parameters(), lr=0.001)
P_loss_fn = nn.MSELoss()

# 馬達實際資料
# path1 = 'sim_data/dataset/x_data_all_15000_Q2.txt'
# path2 = 'sim_data/dataset/P_data_all_15000_Q2.txt'
# path3 = 'sim_data/dataset/raw_data_all_15000_Q2.txt'
path1 = 'sim_data/dataset/AKF/x_data_all_Q12_addNoise_15000.txt'
path2 = 'sim_data/dataset/AKF/P_data_all_Q12_addNoise_15000.txt'
path3 = 'sim_data/dataset/AKF/raw_data_all_Q12_addNoise_15000.txt'
x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, rawt_data_all  = dataset_arrange.loadSimData(path1, path2, path3)
# print("x_input_data_all =", x_input_data_all)
P_true = np.array([[1e-7, 1e-7, 1e-7],
                   [1e-7, 1e-7, 1e-7],
                   [1e-7, 1e-7, 1e-7]])
# P_true = np.array([[1e-7]])
x_y_true_all = []
x_y_pred_all = []
x_loss_data = []
x_rmse_loss_data = []
x_rmse_total_data = []
P_y_true_all = []
P_y_pred_all = []
P_loss_data = []
P_rmse_loss_data = []
P_rmse_total_data = []

total_epoch = epoch
for epoch in range(epoch + 1):
    x_total_loss = 0
    P_total_loss = 0

    # 創建批次數據
    x_input_data = []
    for i in range(0, traning_size, batch_size):
        # print("x_input_data_all=", x_input_data_all)
        batch_x_input_data_all = x_input_data_all[i:i+batch_size] # me
        # 添加到批次列表中
        x_input_data = batch_x_input_data_all# me
        # 將數據轉換為張量，並添加一個維度以符合 LSTM 的輸入格式
        # print("x_input_data=", x_input_data)
        x_input_tensor = torch.tensor(np.vstack(x_input_data), dtype=torch.float32).unsqueeze(1).to(device)
        # LSTM進行狀態估計
        x_lstm_output = x_lstm_model(x_input_tensor)

        # 計算損失
        # x_target = torch.tensor(np.array(x_input_data_all)[1:,:2], dtype=torch.float32).to(device)
        # x_loss = x_loss_fn(x_lstm_output[1:batch_size, :2], x_target[i+1:i+batch_size,:2]) #可以得到一個epoch中每筆資料的mse
        x_target = torch.tensor(np.array(x_k_update_data)[:, :3], dtype=torch.float32).to(device)

        min1 = np.min(np.array(x_k_update_data)[:, 0])
        max1 = np.max(np.array(x_k_update_data)[:, 0])
        norm1 = max1 - min1
        x_lstm_output[:, 0] = (x_lstm_output[:, 0]-min1)/norm1
        x_target[:, 0] = (x_target[:, 0]-min1)/norm1

        min2 = np.min(np.array(x_k_update_data)[:, 1])
        max2 = np.max(np.array(x_k_update_data)[:, 1])
        norm2 = max2 - min2
        x_lstm_output[:, 1] = (x_lstm_output[:, 1]-min2)/norm2
        x_target[:, 1] = (x_target[:, 1]-min2)/norm2

        min3 = np.min(np.array(x_k_update_data)[:, 2])
        max3 = np.max(np.array(x_k_update_data)[:, 2])
        norm3 = max3 - min3
        x_lstm_output[:, 2] = (x_lstm_output[:, 2]-min3)/norm3
        x_target[:, 2] = (x_target[:, 2]-min3)/norm3

        # x_loss0 = x_loss_fn(x_lstm_output[0:batch_size, 0], x_target[i:i+batch_size, 0])
        # x_loss1 = x_loss_fn(x_lstm_output[0:batch_size, 1], x_target[i:i+batch_size, 1])
        # x_loss2 = x_loss_fn(x_lstm_output[0:batch_size, 2], x_target[i:i+batch_size, 2])
        # print(f'[pos_loss:{x_loss0} -- vel_loss:{x_loss1} -- acc_loss:{x_loss2}]')

        x_loss = x_loss_fn(x_lstm_output[0:batch_size, 0:3], x_target[i:i+batch_size, 0:3])
        # x_loss = x_loss_fn(x_lstm_output[0:batch_size, :1], x_target[i:i+batch_size]) #可以得到一個epoch中每筆資料的mse
        x_loss_data.append(x_loss.item()) 
        x_rmse_loss = torch.sqrt(x_loss) #可以得到一個epoch中每筆資料的rmse
        x_rmse_loss_data.append(x_rmse_loss.item())
        x_total_loss += x_rmse_loss.item()

        # 保存真實值和預測值
        x_y_true_all.append(x_true.flatten())
        x_y_pred_all.append(x_lstm_output.detach().cpu().numpy().flatten())
        # print("x_y_pred_all =", x_y_pred_all)

        # 反向傳播和參數更新
        x_optimizer.zero_grad()
        x_loss.backward()
        x_optimizer.step()
# ------------------------------------------------------ #
    # --------狀態估測誤差協方差模型-------- #
    # P_k_update_data = np.array(P_k_update_data)
    # Knp_data = np.array(KCP_data)

    # # 創建批次數據
    # P_input_data = []
    # for i in range(0, traning_size, batch_size):
    #     print("P_input_data_all =", P_input_data_all)
    #     batch_P_input_data_all = P_input_data_all[i:i+batch_size]# me
    #     # 添加到批次列表中
    #     P_input_data = batch_P_input_data_all# me
    
    #     # 將數據轉換為張量，並添加一個維度以符合 LSTM 的輸入格式
    #     P_input_tensor = torch.tensor(np.vstack(P_input_data), dtype=torch.float32).unsqueeze(1).to(device)
    #     # print("P_input_tensor+", P_input_tensor)
    #     # LSTM進行狀態估計
    #     P_lstm_output = P_lstm_model(P_input_tensor)

    #     # 計算損失
    #     # P_target = torch.tensor(np.array(P_input_data_all)[:, :4], dtype=torch.float32).to(device)
    #     # P_loss = P_loss_fn(P_lstm_output[:, :4], P_target[i:i+batch_size, :4]) #可以得到一個epoch中每筆資料的mse
    #     P_target = torch.tensor(np.array(P_input_data_all)[:, :9], dtype=torch.float32).to(device)
    #     P_loss = P_loss_fn(P_lstm_output[:, :9], P_target[i:i+batch_size, :9]) #可以得到一個epoch中每筆資料的mse
    #     P_loss_data.append(P_loss.item()) 
    #     P_rmse_loss = torch.sqrt(P_loss) #可以得到一個epoch中每筆資料的rmse
    #     P_rmse_loss_data.append(P_rmse_loss.item())
    #     P_total_loss += P_rmse_loss.item()

    #     # 保存真實值和預測值
    #     P_y_true_all.append(P_true.flatten())
    #     P_y_pred_all.append(P_lstm_output.detach().npu().numpy().flatten())
    #     # print("P_y_pred_all =", P_y_pred_all)

    #     # 反向傳播和參數更新
    #     P_optimizer.zero_grad()
    #     P_loss.backward()
    #     P_optimizer.step()

    x_rmse_total = np.sqrt(np.mean(np.array(x_rmse_loss_data)**2)) #可以得到每一個epoch的rmse
    x_rmse_total_data.append(x_rmse_total)
    # P_rmse_total = np.sqrt(np.mean(np.array(P_rmse_loss_data)**2)) #可以得到每一個epoch的rmse
    # P_rmse_total_data.append(P_rmse_total)
    if epoch % 1 == 0:
        print(f'-------------------------------------------------------------------')
        print(f'|Epoch : {epoch}/{total_epoch} | x_Loss_RMSE : {x_rmse_total.item():.4f}|')
        # print(f'|Epoch : {epoch}/{total_epoch} | x_Loss_RMSE : {x_rmse_total.item():.4f} | P_Loss_RMSE : {P_rmse_total.item():.4f}|')

# 計算 RMSE
x_y_true_all = np.array(x_y_true_all)
x_y_pred_all = np.array(x_y_pred_all)
P_y_true_all = np.array(P_y_true_all)
P_y_pred_all = np.array(P_y_pred_all)

os.makedirs('motor/motor_model', exist_ok=True)
# x result儲存模型
torch.save(x_lstm_model.state_dict(), 'sim_data/model/x_model.pth')
print("-------- x Model saved successfully --------")
# P result儲存模型
# torch.save(P_lstm_model.state_dict(), 'sim_data/model/P_model.pth')
# print("-------- P Model saved successfully --------")

# --------狀態估測誤差模型-------- #
plt.figure(figsize=(12, 6))
plt.plot(np.array(x_rmse_loss_data), label='loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('x_loss')
plt.legend()
plt.title('x RMSE for every data in each epoch')

plt.figure(figsize=(12, 6))
Epoch = np.arange(1, len(x_rmse_total_data) + 1)
plt.plot(Epoch, np.array(x_rmse_total_data), label='loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('x_loss')
plt.legend()
plt.title('Epoch vs RMSE')

# --------狀態估測誤差協方差模型-------- #
# plt.figure(figsize=(12, 6))
# plt.plot(np.array(P_y_pred_all)[:, 0], label='X1', color='blue')
# plt.plot(np.array(P_y_pred_all)[:, 4], label='X2', color='red')
# plt.plot(np.array(P_y_pred_all)[:, 8], label='X2', color='red')
# plt.xlabel('Time Step')
# plt.ylabel('error cov')
# plt.legend()
# plt.title('error cov iter')

# plt.figure(figsize=(12, 6))
# Epoch = np.arange(1, len(x_rmse_total_data) + 1)
# plt.plot(Epoch, np.array(P_rmse_total_data), label='loss', color='blue')
# plt.xlabel('Epoch')
# plt.ylabel('P_loss')
# plt.legend()
# plt.title('Epoch vs RMSE')

plt.show()
