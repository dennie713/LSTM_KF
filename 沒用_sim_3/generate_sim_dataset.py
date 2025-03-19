import numpy as np
import matplotlib.pyplot as plt

# x = Ax + Bu
# y = Cx

def RawData(t):
    # pos
    A = 2 * np.sin(0.05*t)
    x_p1 = A + 0
    C = 0.5
    y_p = C * x_p1
    x_p = x_p1

    # vel 
    x_v1 = (2 * 0.05 * np.cos(0.05*t)) + 0
    C = 0.5
    y_v = C * x_v1
    x_v = x_v1
 
    # acc
    x_a1 = (- 2 * 0.05**2 * np.sin(0.05*t)) + 0
    C = 0.5
    y_a = C * x_a1
    x_a = x_a1

    x_p1_addNoise = x_p1 + np.random.normal(0, 0.05, np.array(x_p1).shape)
    return A, x_p, x_p1, x_v1, x_a1, x_p1_addNoise

def genRawData(dataset_size):
    A_data = []
    x_data = [0.1]
    pos_data = []
    vel_data = []
    acc_data = []
    x_p1_addNoise_data = []
    for t in range(dataset_size):
        A, x_p, x_p1, x_v1, x_a1, x_p1_addNoise = RawData(t)
        A_data.append(A)
        x_data.append(x_p)
        pos_data.append(x_p1)
        vel_data.append(x_v1)
        acc_data.append(x_a1)
        x_p1_addNoise_data.append(x_p1_addNoise)
    return A_data, x_data, pos_data, vel_data, acc_data, x_p1_addNoise_data

class KalmanFilter:
    def __init__(self, A_data, x_data, x_p1_addNoise_data):
        # 1個系統
        B = np.array([0])
        H = np.array([0.5]) # 1/cpi 
        # 過程噪聲 
        Q = np.array([0.8]) 
        # 測量噪聲
        R = 0.95 #與誤差有關 -> 影響平滑度
        # 誤差協方差矩陣
        P = np.array([1e-4])
        x = np.array([0.1])    
        # 控制輸入
        u = np.zeros((1))
        
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
        # 1個系統   
        self.P_k = np.array([1e-8])
        self.x_true = np.zeros((1, 1))
        self.k_y = np.zeros((1, 1))
        self.KCP = np.zeros((1, 1))
        self.z = np.zeros((1, 1))
        self.z_data = [self.z.flatten()]
        self.k_y_data = [self.k_y.flatten()]
        # self.x_true_data = []
        # self.x_true_data = [self.x.flatten()]
        self.x_true_data = [np.array(x_data)]
        # print('x_true_data =', len(self.x_true_data))
        self.x_true_data_noise = [x_p1_addNoise_data]
        self.KCP_data = [self.KCP.flatten()]
        self.x_k_predict_data = []
        self.P_k_predict_data = []
        self.x_k_update_data = [self.x.flatten()]
        self.P_k_update_data = [self.P.flatten()]
        self.P_k_data = [self.P_k.flatten()]
    
    def KF(self, k, A_data, x_data, x_p1_addNoise_data): 
        # 狀態預測
        # self.x_k = self.x_true_data[k].reshape(2, 1) + self.B @ self.u # x_k為預測的x
        # self.x_k = self.x_true_data[k] + self.B * self.u # x_k為預測的x
        self.x_k = x_data[k] + self.B * self.u # x_k為預測的x
       
        self.P_k = A_data[k] * self.P * A_data[k].T + self.Q # P_k為預測的x
        # self.P_k = A_data[k] @ self.P @ A_data[k].T + self.Q # P_k為預測的x
        self.P_k_data.append(self.P_k.flatten())
        # 卡爾曼增益
        # self.K = self.P_k @ self.H.T @ cp.linalg.inv(self.H @ self.P_k @ self.H.T + self.R)
        self.K = self.P_k * self.H.T * (self.H * self.P_k * self.H.T + self.R)
        # 狀態更新
        # self.z = self.H @ A_data[k] @ self.x_k
        # self.z  = self.x_true_data_noise[k] 
        # self.z  = self.H * self.x_true_data_noise[k]
        self.z  = self.H * x_p1_addNoise_data[k]
        self.z_data.append(self.z)
        y_tel = (self.z - self.H * self.x_k) # z為實際觀測到的
        self.k_y = self.K * y_tel
        self.k_y_data.append(self.k_y)
        self.x_k_1 = self.x_k + self.k_y # self.x_k_1為更新後的x
        self.x = self.x_k_1 # self.x更新後跌代
        self.x_k_update_data.append(self.x.flatten())
        self.x_k_predict_data.append(self.x_k.flatten())

        # 誤差協方差更新
        # self.KCP = self.K @ self.H @ self.P_k
        self.KCP = self.K * self.H * self.P_k
        self.KCP_data.append(self.KCP.flatten())
        self.P_k_1 = (np.eye((self.K * self.H).shape[0]) - self.K * self.H) * self.P_k # self.P_k_1為更新後的x
        # self.P_k_1 = (cp.eye((self.K @ self.H).shape[0]) - self.K @ self.H) @ self.P_k # self.P_k_1為更新後的x
        self.P = self.P_k_1 # self.P更新後跌代
        self.P_k_update_data.append(self.P.flatten())
        self.P_k_predict_data.append(self.P_k.flatten())

        return self.x_k_update_data, self.x_k_predict_data, self.P_k_update_data, self.P_k_predict_data, self.k_y_data, self.KCP_data, self.z_data, self.P_k_data

def KF_Process(dataset_size):
    A_data, x_data, pos_data, vel_data, acc_data, x_p1_addNoise_data = genRawData(dataset_size)
    # print("pos_data_addNoise =", x_p1_addNoise_data)
    # print("x_data =", x_data)  
    KF = KalmanFilter(A_data, x_data, x_p1_addNoise_data)
    for k in range(dataset_size):
        print(k+1)
        x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, P_k_data = KF.KF(k, A_data, x_data, x_p1_addNoise_data)
    return x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, P_k_data 

if __name__ == "__main__":
    dataset_size = 200
    x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, P_k_data = KF_Process(dataset_size)
    A_data, x_data, pos_data, vel_data, acc_data, x_p1_addNoise_data = genRawData(dataset_size)
    
    x_true_data = x_data
    x_true_data_noise = x_p1_addNoise_data
    x_tel = np.array(x_true_data) - np.array(x_k_update_data)
    pos_data = [np.array(item).reshape(-1) for item in pos_data]
    vel_data = [np.array(item).reshape(-1) for item in vel_data]
    acc_data = [np.array(item).reshape(-1) for item in acc_data]
    x_true_data = [np.array(item).reshape(-1) for item in x_true_data]
    x_true_data_noise = [np.array(item).reshape(-1) for item in x_true_data_noise]
    k_y_data = [np.array(item).reshape(-1) for item in k_y_data]
    # x
    x_k_update_data = np.array(x_k_update_data)
    k_y_data = np.array(k_y_data)
    x_tel = np.array(x_tel)
    x_true_data = np.array(x_true_data)
    x_true_data_noise = np.array(x_true_data_noise)
    z_data = np.array(z_data)
    x_k_predict_data = np.array(x_k_predict_data)
    # p
    P_k_update_data = np.array(P_k_update_data)
    Knp_data = np.array(KCP_data)
    
    # print("x_k_update_data.shape =", x_k_update_data.shape)
    # print("x_true_data.shape =", x_true_data.shape)
    # print("k_y_data.shape =", k_y_data.shape)
    #　匯出數據
    # --------狀態估測模型-------- #
    x_data_all = np.concatenate((x_k_update_data[0:dataset_size - 1], k_y_data[0:dataset_size - 1], x_tel[0:dataset_size - 1], x_true_data[0:dataset_size - 1], x_true_data_noise[0:dataset_size - 1], z_data[0:dataset_size - 1], x_k_predict_data[0:dataset_size - 1]), axis=1)# me
    # --------狀態估測誤差協方差模型-------- #
    P_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)# me
    # --------對照數據-------- #
    raw_data_all = np.concatenate((x_true_data_noise[0:dataset_size - 1], pos_data[0:dataset_size - 1], vel_data[0:dataset_size - 1], acc_data[0:dataset_size - 1]), axis=1)# me
    
        
    # 假设你的 x_input_data_all 是 cupy 数组
    # 使用 numpy.savetxt 将其保存到 txt 文件中
    np.savetxt('sim_3/x_data_all.txt', x_data_all, delimiter=' ')
    np.savetxt('sim_3/P_data_all.txt', P_data_all, delimiter=' ')
    np.savetxt('sim_3/raw_data_all.txt', raw_data_all, delimiter=' ')

    # 畫出數據圖
    plt.figure()
    plt.plot(pos_data, label='True_pos', color='black', linewidth=1)
    plt.plot(x_true_data, label='x_true_data', color='purple', linewidth=1)
    plt.plot(x_p1_addNoise_data, label='True_pos_addNoise', color='orange', linewidth=1)
    # plt.plot(x_k_update_data, label='LKF_vel_addNoise', color='cyan', linewidth=1)
    plt.plot(vel_data, label='True_vel', color='red', linewidth=1)
    plt.plot(acc_data, label='True_acc', color='blue', linewidth=1)
    plt.xlabel('t')
    plt.ylabel('value')
    plt.legend()
    plt.title('pos, vel, acc')
    plt.show()