import numpy as np
import cupy as cp

# 讀取 txt 文件
def load_data(path_x, path_p):
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

# x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, P_k_update_data, KCP_data, P_input_data_all, x_k_predict_data = load_data('x_input_data_all_10000.txt', 'P_input_data_all_10000.txt')
# print("x_data =", x_data)
# print("x_k_update_data =", x_k_update_data)
# print("x_input_data_all =", x_input_data_all)
# print("k_y_data =", k_y_data)