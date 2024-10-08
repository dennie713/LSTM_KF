import numpy as np
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import LSTM, KF
# 強制使用CPU
device = torch.device("cpu") 

# 設置模型參數
input_size = 12  # 根據需要的特徵數
hidden_size = 16
output_size = 12   # 假設輸出有兩個維度

# 加載模型
lstm_model_loaded = LSTM.LSTM_KF(input_size, hidden_size, output_size)  # 創建模型實例
lstm_model_loaded.load_state_dict(torch.load('lstm_kf_model.pth', weights_only=True))  # 加載權重
lstm_model_loaded.eval()  # 將模型設置為評估模式

# 準備輸入數據
# 讀取檔案
path1 = ['D:/ASUS_program_code/asus_code_backup\cmake_mouse_boundary_v9_1/build/IPS750_G50_F_motion.txt'] #馬達資料.txt路徑
path2 = ['D:/ASUS_program_code/asus_code_backup\cmake_mouse_boundary_v9_1/build/IPS750_G50_F_mouse.txt']  #滑鼠資料.txt路徑
x_kf_update_data, P_kf_update_data, Pos, PosCmd = KF.KF_Process(path1, path2)
z = Pos
validation_num = len(z)%2 + 1
for k in range(validation_num, len(z)):
    input_data = torch.tensor(np.hstack([x_kf_update_data[k], P_kf_update_data[k]]), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # 假設我們要傳遞一個隨機生成的輸入，形狀應該是 (batch_size, sequence_length, input_size)
    # batch_size = 1
    # sequence_length = 1
    # input_data = np.random.rand(batch_size, sequence_length, input_size).astype(np.float32)  # 隨機生成數據
    input_tensor = torch.tensor(input_data)  # 轉換為 PyTorch 張量

# 使用模型進行推斷
with torch.no_grad():  # 禁用梯度計算以提高推斷效率
    lstm_output = lstm_model_loaded(input_tensor)  # 獲取模型的輸出

print("LSTM Output:", lstm_output.numpy())  # 輸出結果