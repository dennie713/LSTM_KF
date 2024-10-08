import numpy as np
import load_test
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

x_data, x_k_update_data, k_y_data, x_tel, x_true, x_true_noise, x_obsve, x_input_data_all, x_k_predict_data, P_data, P_k_update_data, KCP_data, P_input_data_all = load_test.load_data('x_data_all_15000.txt', 'P_data_all_15000.txt')
test = [[1, 2, 3, 4],
        [5, 6, 7, 8]]
test = pd.DataFrame(test)
x_true = pd.DataFrame(x_true)
x_input_data_all = pd.DataFrame(x_input_data_all)
P_input_data_all = pd.DataFrame(P_input_data_all)
# 歸一化
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler.fit(x_input_data_all)
# scaler.fit(P_input_data_all)
# x_input_data_all_normalized = scaler.fit_transform(x_input_data_all)
# P_input_data_all_normalized = scaler.fit_transform(x_input_data_all)

# 初始化 MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# 轉置 DataFrame，然後進行歸一化
scaler.fit(test)
test_normalized = pd.DataFrame(scaler.fit_transform(test), columns=test.columns)
scaler.fit(x_input_data_all)
x_input_data_all_normalized = pd.DataFrame(scaler.fit_transform(x_input_data_all), columns=x_input_data_all.columns)
scaler.fit(P_input_data_all)
P_input_data_all_normalized = pd.DataFrame(scaler.fit_transform(P_input_data_all), columns=P_input_data_all.columns)
scaler.fit(x_true)
x_true_normalized = pd.DataFrame(scaler.fit_transform(x_true), columns=x_true.columns)
# # 標準化
# scaler = StandardScaler()
# data_standardized = scaler.fit_transform(data)

# 使用 numpy.savetxt 将其保存到 txt 文件中
np.savetxt('test_normalized.txt', test_normalized, delimiter=' ')
np.savetxt('x_input_data_all_normalized.txt', x_input_data_all_normalized, delimiter=' ')
np.savetxt('P_input_data_all_normalized.txt', P_input_data_all_normalized, delimiter=' ')
np.savetxt('x_true_normalized.txt', x_true_normalized, delimiter=' ')