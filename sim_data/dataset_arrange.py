import numpy as np

# 讀取 txt 文件
def loadSimData(path_x, path_p, path_raw):

    # raw data順序true_pos, true_vel, true_acc
    raw_data = np.loadtxt(path_raw, delimiter=' ')
    true_pos = raw_data[:, 0]
    true_vel = raw_data[:, 1]
    true_acc = raw_data[:, 2]
    raw_data_all = raw_data 

    # x
    x_data = np.loadtxt(path_x, delimiter=' ') #path = 'x_input_data_all.txt'
    # print(x_data.shape)
    # 順序x_k_update_data, k_y_data, x_tel, x_true_data, x_true_data_noise
    x_k_update_data = x_data[:, 0:3]
    k_y_data = x_data[:, 3:6]
    x_tel = x_data[:, 6:9]
    # prediction_errors_data = x_data[:, 6:8]
    x_true = x_data[:, 9:10] # x_true_data
    x_true_noise = x_data[:, 10:11] # x_true_data_noise
    # x_obsve = x_data[:, 10]# z_data
    # x_k_predict_data = x_data[:, 11:13]
    x_input_data_all = np.concatenate((x_true_noise, x_k_update_data, raw_data), axis=1)
    # x_input_data_all = np.concatenate((x_true_noise, k_y_data, x_tel), axis=1)
    # print(x_input_data_all)

    # p
    P_data = np.loadtxt(path_p, delimiter=' ') # 'P_data_10000.txt'
    # data排列順序P_k_update_data, KCP_data
    P_k_update_data = P_data[:, 0:9]
    KCP_data = P_data[:, 9:18]
    P_input_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)
    # print("P_input_data_all =", P_input_data_all)

    return x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_input_data_all, P_data, P_k_update_data, KCP_data, P_input_data_all, raw_data_all 
