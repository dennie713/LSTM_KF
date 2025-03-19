import numpy as np
import cupy as cp

# 讀取 txt 文件
def loadSimData(path_x, path_p):
    # x
    x_data = np.loadtxt(path_x, delimiter=' ') #path = 'x_input_data_all.txt'
    # print(x_data.shape)
    # 順序x_k_update_data, k_y_data, x_tel, x_true_data, x_true_data_noise, z_data, x_k_predict_data
    x_k_update_data = x_data[:, 0:2]
    k_y_data = x_data[:, 2:4]
    x_tel = x_data[:, 4:6]
    # prediction_errors_data = x_data[:, 6:8]
    x_true = x_data[:, 6:8] # x_true_data
    x_true_noise = x_data[:, 8:10] # x_true_data_noise
    x_obsve = x_data[:, 10]# z_data
    x_k_predict_data = x_data[:, 11:13]
    x_input_data_all = np.concatenate((x_k_update_data, k_y_data, x_tel), axis=1)
    # print(x_input_data_all)

    # p
    P_data = np.loadtxt(path_p, delimiter=' ') # 'P_data_10000.txt'
    # data排列順序P_k_update_data, KCP_data
    P_k_update_data = P_data[:, 0:4]
    KCP_data = P_data[:, 4:8]
    P_input_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)
    return x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, x_k_predict_data, P_data, P_k_update_data, KCP_data, P_input_data_all

def loadMotorData(path_x, path_P):
    # Pos, pose, vele, acce, PosCmd, VelCmd, AccCmd, km_y_data
    # 馬達實際資料
    x_data = np.loadtxt(path_x) # motor_dataset/Motor_x_data.txt
    P_data = np.loadtxt(path_P) # motor_dataset/Motor_P_data.txt
    # Pos, pose, vele, acce, km_y_data, x_tel_data
    x_true = x_data[:,0]
    x_k_update_data = x_data[:, 1:4]
    x_cmd = x_data[:, 4:7]
    km_y_data = x_data[:, 7]
    x_tel = x_cmd - x_k_update_data
    x_input_data_all = np.concatenate((x_k_update_data, km_y_data, x_tel), axis = 1)
    # Pm_data, kcp_data
    P_k_update_data = P_data[:, :9]
    KCP_data = P_data[:, 9:17]
    P_input_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)
    return x_data, x_true, x_k_update_data, x_cmd, km_y_data, x_tel, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all


def loadSimData_less_feature(path_x, path_p):
    # x
    x_data = np.loadtxt(path_x, delimiter=' ') #path = 'x_input_data_all.txt'
    # print(x_data.shape)
    # 順序x_k_update_data, k_y_data, x_tel, x_true_data, x_true_data_noise, z_data, x_k_predict_data
    x_k_update_data = x_data[:, 0:2]
    k_y_data = x_data[:, 2:4]
    x_tel = x_data[:, 4:6]
    # prediction_errors_data = x_data[:, 6:8]
    x_true = x_data[:, 6:8] # x_true_data
    x_true_noise = x_data[:, 8:10] # x_true_data_noise
    x_obsve = x_data[:, 10]# z_data
    x_k_predict_data = x_data[:, 11:13]
    x_input_data_all = x_k_update_data
    # x_input_data_all = np.concatenate((x_k_update_data, k_y_data, x_tel), axis=1)
    # print(x_input_data_all)

    # p
    P_data = np.loadtxt(path_p, delimiter=' ') # 'P_data_10000.txt'
    # data排列順序P_k_update_data, KCP_data
    P_k_update_data = P_data[:, 0:4]
    KCP_data = P_data[:, 4:8]
    P_input_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)
    return x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, x_k_predict_data, P_data, P_k_update_data, KCP_data, P_input_data_all

def loadSimData_1_feature(path_x, path_p):
    # x
    x_data = np.loadtxt(path_x, delimiter=' ') #path = 'x_input_data_all.txt'
    # print(x_data.shape)
    # 順序x_k_update_data, k_y_data, x_tel, x_true_data, x_true_data_noise, z_data, x_k_predict_data
    x_k_update_data = x_data[:, 0:2]
    k_y_data = x_data[:, 2:4]
    x_tel = x_data[:, 4:6]
    # prediction_errors_data = x_data[:, 6:8]
    x_true = x_data[:, 6:8] # x_true_data
    x_true_noise = x_data[:, 8:10] # x_true_data_noise
    x_obsve = x_data[:, 10]# z_data
    x_k_predict_data = x_data[:, 11:13]
    x_input_data_all = x_true_noise
    # x_input_data_all = np.concatenate((x_k_update_data, k_y_data, x_tel), axis=1)
    # print(x_input_data_all)

    # p
    P_data = np.loadtxt(path_p, delimiter=' ') # 'P_data_10000.txt'
    # data排列順序P_k_update_data, KCP_data
    P_k_update_data = P_data[:, 0:4]
    KCP_data = P_data[:, 4:8]
    P_input_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)
    return x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, x_k_predict_data, P_data, P_k_update_data, KCP_data, P_input_data_all
