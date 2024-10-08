import cupy as cp
import numpy as np
 

def getMonteCarlo():
    P_k = np.array([[1e-4, 1e-5],
                    [1e-5, 1e-4]])
    # 設定樣本數量
    num_samples = 15000
    # 從多變量正態分佈中生成樣本
    prediction_errors_data = []
    prediction_errors = np.random.multivariate_normal(mean=[0, 0], cov=P_k, size=num_samples)
    # for i in range(len(self.P_k_data)):
    #     prediction_errors = cp.random.multivariate_normal(mean=[0, 0], cov=self.P_k_data[i].reshape(2,2), size=num_samples)
    #     prediction_errors_data.append(prediction_errors)

    return prediction_errors