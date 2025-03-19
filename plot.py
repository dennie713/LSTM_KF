import numpy as np
import motor.ImportData as ImportData, motor.Cal as Cal
import AddNoise
import matplotlib.pyplot as plt

import numpy as np
import motor.ImportData as ImportData, motor.Cal as Cal
import AddNoise
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, B, H, Q, R, P, u, x):
        # self.A = A  # 狀態轉移矩陣
        self.B = B  # 控制矩陣
        self.H = H  # 觀測矩陣
        self.Q = Q  # 過程噪聲
        self.R = R  # 測量噪聲
        self.P = P  # 誤差協方差矩陣
        # self.z = z  # 實際量測值
        self.u = u
        self.x = x # 初始狀態
        self.K = 0
        self.x_k = 0
        self.P_k = np.zeros((2, 2))
        self.x_true = np.zeros((2, 1))
        self.k_y = np.zeros((2, 1))
        self.KCP = np.zeros((2, 2))
        self.z_data = []
        self.k_y_data = [self.k_y.flatten()]
        # self.x_true_data = []
        self.x_true_data = [self.x.flatten()]
        self.x_true_data_noise = []
        self.KCP_data = [self.KCP.flatten()]
        self.x_k_predict_data = []
        self.P_k_predict_data = []
        self.x_k_update_data = [self.x.flatten()]
        self.P_k_update_data = [self.P.flatten()]
        self.P_k_data = [self.P_k.flatten()]
    
    def KF(self, k, A_data): 
        # self.x_true_data = self.x_true_data_flatten + np.random.normal(0, 0.001, np.array(self.x_true_data_flatten).shape)
        # 狀態預測
        # self.x_k = A_data[k] @ self.x + self.B @ self.u # x_k為預測的x
        self.x_k = self.x_true_data_noise[k].reshape(2, 1) + self.B @ self.u # x_k為預測的x
        # print("self.x_k =", self.x_k)
        # self.P_k = A_data[k] @ self.P @ A_data[k].T + self.Q # P_k為預測的x
        self.P_k = A_data[k] @ self.P @ A_data[k].T + self.Q # P_k為預測的x
        self.P_k_data.append(self.P_k.flatten())
        # 卡爾曼增益
        self.K = self.P_k @ self.H.T @ np.linalg.inv(self.H @ self.P_k @ self.H.T + self.R)
        # print("self.K =", self.K)
        # 狀態更新
        # self.z = self.H @ A_data[k] @ self.x_k
        self.z  = self.H @ self.x_true_data_noise[k]
        self.z_data.append(self.z)
        y_tel = (self.z - self.H @ self.x_k) # z為實際觀測到的
        self.k_y = self.K @ y_tel
        self.k_y_data.append(self.k_y)
        self.x_k_1 = self.x_k + self.k_y # self.x_k_1為更新後的x
        self.x = self.x_k_1 # self.x更新後跌代
        # print(self.x.flatten())
        self.x_k_update_data.append(self.x.flatten())
        self.x_k_predict_data.append(self.x_k.flatten())

        # 誤差協方差更新
        self.KCP = self.K @ self.H @ self.P_k
        # print("self.KCP =", self.KCP)
        self.KCP_data.append(self.KCP.flatten())
        # self.P = np.eye(self.A.shape[0]) @ self.P - KCP
        self.P_k_1 = (np.eye((self.K @ self.H).shape[0]) - self.K @ self.H) @ self.P_k # self.P_k_1為更新後的x
        self.P = self.P_k_1 # self.P更新後跌代
        self.P_k_update_data.append(self.P.flatten())
        self.P_k_predict_data.append(self.P_k.flatten())

        return self.x_k_update_data, self.x_k_predict_data, self.P_k_update_data, self.P_k_predict_data, self.k_y_data, self.KCP_data, self.z_data, self.P_k_data
    
    def get_real_x(self, k, A_data):
        # print(k)
        x = np.array([[0.2],
                      [0.5]]) 
        self.x_true = A_data[k] @ x # 實際值
        # print("self.x_true =", self.x_true)
        x = self.x_true
        # self.x_true_data.append(self.x_true.tolist())
        self.x_true_data.append(self.x_true.flatten())
        self.x_true_data_noise = self.x_true_data + np.random.normal(0, 0.003, np.array(self.x_true_data).shape)
        # print(self.x_true_data_noise[1].reshape(2, 1).shape)
        return self.x_true_data, self.x_true_data_noise
    
    def getMonteCarlo(self):
        # 設定樣本數量
        num_samples = 1
        # 從多變量正態分佈中生成樣本
        prediction_errors_data = []
        for i in range(len(self.P_k_data)):
            prediction_errors = np.random.multivariate_normal(mean=[0, 0], cov=self.P_k_data[i].reshape(2,2), size=num_samples)
            prediction_errors_data.append(prediction_errors)
        # prediction_errors_data = np.array(prediction_errors_data).reshape(10000, 2)
        return prediction_errors_data

def KF_Process(k):
    A_data = []
    # 狀態轉移矩陣
    for k in range(k):
        # print(k)
        k = k 
        A = np.array([[0.7 + 0.05 * np.sin(0.001*k), 0.06 + 0.04 * np.arctan(1 / (k + 1))],
                    [0.10 - 0.1 * np.sin(0.001*k), -0.2 + 0.2 * np.sin(0.001*k)]])
        A_data.append(A)
    # 控制矩陣
    B = np.array([[0],
                  [0]])
    H = np.array([[1.8, 2]]) # 1/cpi 
    # 過程噪聲 
    Q = np.array([[0.8, 0],
                  [0, 1.2]]) 
    # 測量噪聲
    R = 0.95 #與誤差有關 -> 影響平滑度
    # 誤差協方差矩陣
    P = np.array([[1e-4, 1e-4],
                  [1e-4, 1e-4]])
    x = np.array([[0.2],
                  [0.5]]) 
        
    # 控制輸入
    u = np.zeros((1, 1))

    # 創建卡爾曼濾波器實例
    KF = KalmanFilter(B, H, Q, R, P, u, x)
    for k in range(k):
        # 卡爾曼濾波器更新步驟
        x_true_data, x_true_data_noise = KF.get_real_x(k, A_data)
        x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, P_k_data = KF.KF(k, A_data)
        prediction_errors_data = KF.getMonteCarlo()
    return x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, x_true_data, x_true_data_noise, P_k_data, prediction_errors_data

x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, x_true_data, x_true_data_noise, P_k_data, prediction_errors_data = KF_Process(1000)
print("prediction_errors_data =", prediction_errors_data)


plt.figure()
start_size = 10000
validation_size = 5000
plt.plot(np.array(x_true_data_noise)[1:,0], label='True_+noise_x1', color='red', linewidth=1)
plt.plot(np.array(x_true_data_noise)[1:,1], label='True_+noise_x2', color='red', linewidth=1)
plt.plot(np.array(x_k_update_data)[1:,0], label='LKF_x1', color='orange', linewidth=1)
plt.plot(np.array(x_k_update_data)[1:,1], label='LKF_x2', color='cyan', linewidth=1)
plt.plot(np.array(x_true_data)[1:,0], label='True_x1', color='black', linewidth=2)
plt.plot(np.array(x_true_data)[1:,1], label='True_x2', color='blue', linewidth=2)
# plt.plot(z_data[:,0], label='z', color='red', linewidth=1)
plt.xlabel('data')
plt.ylabel('value')
plt.legend()
plt.title('estimate vs true :x1 x2')
# plt.show()

# 蒙地卡羅誤差
import numpy as np
import matplotlib.pyplot as plt


print(np.array(prediction_errors_data).shape)
print(np.array(prediction_errors_data).reshape(1000, 2))

# 繪製生成的預測誤差
plt.figure(figsize=(8, 8))
for i in range(len(P_k_data)):
    plt.scatter(np.array(prediction_errors_data)[i, 0], np.array(prediction_errors_data)[i, 1], alpha=0.5)
plt.title('Monte Carlo Sampling of Prediction Errors')
plt.xlabel('Error in State 1')
plt.ylabel('Error in State 2')
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.grid()
plt.axis('equal')
plt.show()

    # # 計算預測誤差的均值和標準差
    # mean_error = np.mean(prediction_errors, axis=0)
    # std_error = np.std(prediction_errors, axis=0)

    # print(f"Mean Prediction Error: {mean_error}")
    # print(f"Standard Deviation of Prediction Error: {std_error}")
