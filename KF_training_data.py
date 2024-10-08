import numpy as np
import ImportData, Cal
import AddNoise
import matplotlib.pyplot as plt
import cupy as cp
# import numpy as cp
import ImportData, Cal
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
        self.P_k = np.array([[1e-8, 1e-8],
                             [1e-8, 1e-8]])
        self.x_true = cp.zeros((2, 1))
        self.k_y = cp.zeros((2, 1))
        self.KCP = cp.zeros((2, 2))
        self.z = cp.zeros((1, 1))
        self.z_data = [self.z.flatten()]
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
        # self.x_true_data = self.x_true_data_flatten + cp.random.normal(0, 0.001, cp.array(self.x_true_data_flatten).shape)
        # 狀態預測
        # self.x_k = A_data[k] @ self.x + self.B @ self.u # x_k為預測的x
        self.x_k = self.x_true_data[k].reshape(2, 1) + self.B @ self.u # x_k為預測的x
       
        # self.P_k = A_data[k] @ self.P @ A_data[k].T + self.Q # P_k為預測的x
        self.P_k = A_data[k] @ self.P @ A_data[k].T + self.Q # P_k為預測的x
        self.P_k_data.append(self.P_k.flatten())
        # 卡爾曼增益
        self.K = self.P_k @ self.H.T @ cp.linalg.inv(self.H @ self.P_k @ self.H.T + self.R)
        # print("self.K =", self.K)
        # 狀態更新
        # self.z = self.H @ A_data[k] @ self.x_k
        # self.z  = self.x_true_data_noise[k] 
        self.z  = self.H @ self.x_true_data_noise[k]
        self.z_data.append(self.z)
        y_tel = (self.z - self.H @ self.x_k) # z為實際觀測到的
        self.k_y = self.K @ y_tel
        # print("self.K = ", self.K)
        # print("y_tel = ", y_tel)
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
        # self.P = cp.eye(self.A.shape[0]) @ self.P - KCP
        self.P_k_1 = (cp.eye((self.K @ self.H).shape[0]) - self.K @ self.H) @ self.P_k # self.P_k_1為更新後的x
        self.P = self.P_k_1 # self.P更新後跌代
        self.P_k_update_data.append(self.P.flatten())
        self.P_k_predict_data.append(self.P_k.flatten())

        return self.x_k_update_data, self.x_k_predict_data, self.P_k_update_data, self.P_k_predict_data, self.k_y_data, self.KCP_data, self.z_data, self.P_k_data
    
    def get_real_x(self, k, A_data):
        # print(k)
        x = cp.array([[0.2],
                      [0.5]]) 
        self.x_true = A_data[k] @ x # 實際值
        # print("self.x_true =", self.x_true)
        x = self.x_true
        # self.x_true_data.append(self.x_true.tolist())
        # self.x_true_data = cp.array(self.x_true_data)
        # self.x_true_data = cp.array(self.x_true_data)
        self.x_true_data = cp.vstack([self.x_true_data, self.x_true.flatten()])
        # self.x_true_data.append(self.x_true.flatten())
        # 實際值加入雜訊
        self.x_true_data_noise = self.x_true_data + cp.random.normal(0, 0.05, cp.array(self.x_true_data).shape)
        # print(self.x_true_data_noise[1].reshape(2, 1).shape)
        return self.x_true_data, self.x_true_data_noise
    
    def getMonteCarlo(self):
        # 設定樣本數量
        num_samples = 10000
        # 從多變量正態分佈中生成樣本
        prediction_errors_data = []
        prediction_errors = cp.random.multivariate_normal(mean=[0, 0], cov=self.P_k.reshape(2,2), size=num_samples)
        # for i in range(len(self.P_k_data)):
        #     prediction_errors = cp.random.multivariate_normal(mean=[0, 0], cov=self.P_k_data[i].reshape(2,2), size=num_samples)
        #     prediction_errors_data.append(prediction_errors)
        # prediction_errors_data = cp.array(prediction_errors_data).reshape(10000, 2)
        return prediction_errors_data

def KF_Process(k):
    A_data = []
    # 狀態轉移矩陣
    for k in range(k):
        # print(k)
        k = k 
        A = cp.array([[0.7 + 0.05 * cp.sin(0.001*k), 0.06 + 0.04 * cp.arctan(1 / (k + 1))],
                    [0.10 - 0.1 * cp.sin(0.001*k), -0.2 + 0.2 * cp.sin(0.001*k)]])
        A_data.append(A)
    # 控制矩陣
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

    # 創建卡爾曼濾波器實例
    KF = KalmanFilter(B, H, Q, R, P, u, x)
    for k in range(k):
        print(k+1)
        # print("12312313")
        # 卡爾曼濾波器更新步驟
        x_true_data, x_true_data_noise = KF.get_real_x(k, A_data)
        x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, P_k_data = KF.KF(k, A_data)
        # prediction_errors_data = KF.getMonteCarlo()
    return x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, x_true_data, x_true_data_noise, P_k_data


# 產生數據並儲存.txt
training_size = 15001
data_set_size = training_size
x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, x_true_data, x_true_data_noise, P_k_data = KF_Process(training_size)
# print("x_k_predict_data.shape =", cp.array(x_k_predict_data).shape)
# print("k_y_data =", k_y_data)
# print("z_data =", z_data)
# for k in range(1, training_size):
# 狀態估測與狀態估測誤差協方差要分開訓練
# --------狀態估測模型-------- #
x_tel = x_true_data - cp.array(x_k_update_data)
k_y_data = [cp.array(item).reshape(-1) for item in k_y_data]
# prediction_errors_data = cp.array(prediction_errors_data).reshape(training_size, 2)
x_k_update_data = cp.array(x_k_update_data)
k_y_data = cp.array(k_y_data)
x_tel = cp.array(x_tel)
x_true_data = cp.array(x_true_data)
x_true_data_noise = cp.array(x_true_data_noise)
z_data = cp.array(z_data)
x_k_predict_data = cp.array(x_k_predict_data)
# print("x_k_update_data.shape =", x_k_update_data[0:training_size - 1, :].shape)
# print("k_y_data =", k_y_data[0:training_size - 1, :].shape)
x_data_all = cp.concatenate((x_k_update_data[0:training_size - 1, :], k_y_data[0:training_size - 1, :], x_tel[0:training_size - 1, :], x_true_data[0:training_size - 1, :], x_true_data_noise[0:training_size - 1, :], z_data[0:training_size - 1, :], x_k_predict_data[0:training_size - 1, :]), axis=1)# me

# --------狀態估測誤差協方差模型-------- #
P_k_update_data = cp.array(P_k_update_data)
KCP_data = cp.array(KCP_data)
P_data_all = cp.concatenate((P_k_update_data, KCP_data), axis=1)# me
    
# 假设你的 x_input_data_all 是 cupy 数组
# 先将 cupy 数组转换为 numpy 数组
x_data_all_np = x_data_all.get()
P_data_all_np = P_data_all.get()
# 使用 numpy.savetxt 将其保存到 txt 文件中
np.savetxt('x_data_all.txt', x_data_all_np, delimiter=' ')
np.savetxt('P_data_all.txt', P_data_all_np, delimiter=' ')

plt.figure()
plt.plot(x_true_data[:, 0].get(), label='True_x1', color='black', linewidth=3)
plt.plot(x_true_data[:, 1].get(), label='True_x2', color='blue', linewidth=3)
plt.plot(x_true_data_noise[:, 0].get(), label='True_x1_add_noise', color='purple', linewidth=3)
plt.plot(x_true_data_noise[:, 1].get(), label='True_x2_add_noise', color='red', linewidth=3)
plt.plot(cp.array(x_k_update_data)[:, 0].get(), label='LKF_x1', color='orange', linewidth=2)
plt.plot(cp.array(x_k_update_data)[:, 1].get(), label='LKF_x2', color='cyan', linewidth=2)
# plt.plot(z_data[:,0], label='z', color='red', linewidth=1)
plt.xlabel('data')
plt.ylabel('value')
plt.legend()
plt.title('estimate vs true :x1 x2')
plt.show()
