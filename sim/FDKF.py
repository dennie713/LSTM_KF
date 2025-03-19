# import numpy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import dataset_arrange
import time

class FusionDKF:
    def __init__(self, P_K, P_D, x_K, x_D):
        self.P_K = P_K
        self.P_D = P_D
        self.x_K = x_K
        self.x_D = x_D
        self.H = np.block([[np.eye(2)],
                           [np.eye(2)]])
        self.R = np.block([[np.zeros((2, 2)), np.eye(2)],
                           [np.eye(2), np.zeros((2, 2))]])
        self.z = np.block([[np.zeros((2, 1))],
                           [np.zeros((2, 1))]])
        # print('self.H =', self.H)
        # print('self.H.shape =', self.H.shape)
        # print('self.R =', self.R)
        # print('self.R.shape =', self.R.shape)
        # print('self.z =', self.z)
        # print('self.z.shape =', self.z.shape)
        
        self.x_hat_F = np.zeros((2, 1))
        self.x_hat_F_data = []
        self.P_F = np.zeros((2, 2))
        self.P_F_data = []
    
    def FDKF(self):
        # print("self.P_K =", self.P_K[10000:,:])
        # print("self.P_K.shape =", self.P_K[10000:,:].shape)
        # print("self.P_D =", self.P_D)
        # print("self.P_D.shape =", self.P_D.shape)
        # start_time = time.time()
        for i in range(0, len(P_D)):
            # print(i)
            self.R = np.block([[self.P_K[10000 + i,:].reshape(2, 2), np.zeros((2, 2))],
                               [np.zeros((2, 2)), self.P_D[i,:].reshape(2, 2)]])
            self.z = np.block([[self.x_K[10000 + i,:].reshape(2, 1)],
                               [self.x_D[i,:].reshape(2, 1)]])
            self.x_hat_F = np.linalg.inv(self.H.T @ np.linalg.inv(self.R) @ self.H) @ self.H.T @ np.linalg.inv(self.R) @ self.z
            self.x_hat_F_data.append(self.x_hat_F.flatten())
            # print("self.x_hat_F =", self.x_hat_F)
            # print("self.P_K[9999 + i,:] =", self.P_K[9999 + i,:])
            # print("self.P_D[i,:] =", self.P_D[i,:])
            self.P_F = np.linalg.inv(np.linalg.inv(self.P_K[10000 + i,:].reshape(2, 2)) + np.linalg.inv(self.P_D[i].reshape(2, 2)))
            self.P_F_data.append(self.P_F.flatten())
            # self.x = np.linalg.inv(self.H.T @ np.linalg.inv(self.R) @ self.H) @ self.H.T @ np.linalg.inv(self.R) @ (self.z - self.v)
        return self.x_hat_F_data, self.P_F_data

# FusionDKF = FusionDKF(P_K, P_D, x_K, x_D)
if __name__ == "__main__":
    # LKF資料輸入
    path1 = './sim_dataset/x_data_all_15000_0.001.txt'
    path2 = './sim_dataset/P_data_all_15000_0.001.txt'
    x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, x_k_predict_data, P_data, P_k_update_data, KCP_data, P_input_data_all = dataset_arrange.loadSimData(path1, path2)
    # DKF資料輸入
    path3 = './sim/sim_result/x_lstm_output_data_sim_0.001_layer1.txt'
    path4 = './sim/sim_result/P_lstm_output_data_sim_0.001_layer1.txt'
    x_D_update_data = np.loadtxt(path3, delimiter=' ')
    P_D_update_data = np.loadtxt(path4, delimiter=' ')
    # P_D_update_data = path3
    # x_D_update_data = path4
    # 資料輸入
    P_K = P_k_update_data
    P_D = P_D_update_data
    x_K = x_k_update_data
    x_D = x_D_update_data
    start_time = time.time()
    Fusion_DKF = FusionDKF(P_K, P_D, x_K, x_D)
    x_hat_F_data, P_F_data = Fusion_DKF.FDKF()
    end_time = time.time()
    # for i in range(len(path1)):
    #     # if i == 0:
    #     #     P_K = P_k_update_data[i].reshape(2, 2) + np.array([[0.001, 0],[0, 0]])
    #     P_K = P_k_update_data[i].reshape(2, 2)
    #     P_D = P_D_update_data[i].reshape(2, 2)
    #     x_K = x_k_update_data[i].reshape(2, 1)
    #     x_D = x_D_update_data[i].reshape(2, 1)
    #     Fusion_DKF = FusionDKF(P_K, P_D, x_K, x_D)
    #     x_hat_F_data, P_F_data = Fusion_DKF.FDKF()
    # print('x_hat_F_data.len :', len(x_hat_F_data))

    print('P_F_data :', P_F_data[-1])
    print('x_hat_F_data[0] =', np.array(x_hat_F_data).reshape(-1, 2)[2, 0])
    print("FDKF執行時間 :", end_time - start_time)
    print("FDKF平均一筆執行時間 :", (end_time - start_time)/len(P_D))

    plt.figure()
    plt.plot(np.array(x_hat_F_data).reshape(-1, 2)[:, 0], label='FDKF_x1', color='green', linewidth=1)
    plt.plot(np.array(x_hat_F_data).reshape(-1, 2)[:, 1], label='FDKF_x2', color='yellow', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('fusion value')
    plt.legend()
    plt.title('estimate x1 & x2')

    # 估測狀態匯出
    start_size = 9999
    validation_size = 5000
    plt.figure()
    # x_true_noise = x_true
    plt.plot(x_true_noise[start_size:start_size + validation_size, 0], label='True_x1_add_noise', color='black', linewidth=3)
    plt.plot(x_true_noise[start_size:start_size + validation_size, 1], label='True_x2_add_noise', color='blue', linewidth=3)
    plt.plot(np.array(x_k_update_data)[start_size:start_size + validation_size, 0], label='LKF_x1', color='orange', linewidth=2)
    plt.plot(np.array(x_k_update_data)[start_size:start_size + validation_size, 1], label='LKF_x2', color='cyan', linewidth=2)
    plt.plot(np.array(x_hat_F_data).reshape(-1, 2)[:, 0], label='FDKF_x1', color='green', linewidth=1)
    plt.plot(np.array(x_hat_F_data).reshape(-1, 2)[:, 1], label='FDKF_x2', color='yellow', linewidth=1)
    plt.plot(np.array(x_D_update_data).reshape(-1, 2)[:, 0], label='DKF_x1', color='purple', linewidth=1)
    plt.plot(np.array(x_D_update_data).reshape(-1, 2)[:, 1], label='DKF_x2', color='red', linewidth=1)
    
    plt.xlabel('data')
    plt.ylabel('value')
    plt.legend()
    plt.title('estimate vs true :x1 x2')

    # 估測狀態誤差匯出
    plt.figure()
    # x_k_update_data = cp.array(x_k_update_data).reshape(-1, 1)
    # print("x_k_update_data =", x_k_update_data)
    # print("x_true =", x_true)
    a = np.abs(np.array(x_k_update_data)[start_size:start_size + validation_size, 0] - np.array(x_true_noise)[start_size:start_size + validation_size, 0])
    plt.plot(a, label='LKF_x1', color='orange', linewidth=1)
    b = np.abs(np.array(x_k_update_data)[start_size:start_size + validation_size, 1] - np.array(x_true_noise)[start_size:start_size + validation_size, 1])
    plt.plot(b, label='LKF_x2', color='cyan', linewidth=1)
    # e = np.abs(np.array(x_k_update_data)[start_size:start_size + validation_size, 2] - np.array(x_true_noise)[start_size:start_size + validation_size, 2])
    # plt.plot(e, label='LKF_x2', color='purple', linewidth=2)
    
    # f = cp.abs(cp.array(x_lstm_output_data)[:, 2] - cp.array(x_true_noise)[start_size:start_size + validation_size, 2])
    # plt.plot(f, label='DKF_x2', color='red', linewidth=1)
    e = np.abs(np.array(x_hat_F_data)[:, 0] - np.array(x_true_noise)[start_size:start_size + validation_size, 0])
    plt.plot(e, label='FDKF_x1', color='green', linewidth=1)
    f = np.abs(np.array(x_hat_F_data)[:, 1] - np.array(x_true_noise)[start_size:start_size + validation_size, 1])
    plt.plot(f, label='FDKF_x2', color='yellow', linewidth=1)

    c = np.abs(np.array(x_D_update_data)[:, 0] - np.array(x_true_noise)[start_size:start_size + validation_size, 0])
    plt.plot(c, label='DKF_x1', color='purple', linewidth=1)
    d = np.abs(np.array(x_D_update_data)[:, 1] - np.array(x_true_noise)[start_size:start_size + validation_size, 1])
    plt.plot(d, label='DKF_x2', color='red', linewidth=1)
    plt.xlabel('data')
    plt.ylabel('estimate value')
    plt.legend()
    plt.title('estimate pos vel acc')

    print("---------------------------------------")
    print("LKF_x1 mean error :", np.mean(a))
    print("LKF_x2 mean error :", np.mean(b))
    print("DKF_x1 mean error :", np.mean(c))
    print("DKF_x2 mean error :", np.mean(d))
    print("FDKF_x1 mean error :", np.mean(e))
    print("FDKF_x2 mean error :", np.mean(f))
    print("---------------------------------------")

    plt.show()
    