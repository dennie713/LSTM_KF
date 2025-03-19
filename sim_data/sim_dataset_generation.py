import numpy as np
import matplotlib.pyplot as plt
import LAE, CFD, LSF, KF_v2, zero_phase_filter

class KalmanFilter:
    def __init__(self, dt, Q, R, initial_state=None, initial_covariance=None):
        self.dt = dt
        
        # 狀態轉移矩陣
        self.A_mat = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])

        # 觀測矩陣
        self.C = np.array([[1, 0, 0]])  # 測量僅觀測位置

        # 系統噪聲與測量噪聲
        self.Q = Q
        self.R = R

        # 初始條件
        self.P = np.eye(3) * 1e-3 if initial_covariance is None else initial_covariance
        self.x = np.zeros((3, 1)) if initial_state is None else initial_state
        self.x_k1 = np.zeros((3, 1)) 
        # print("self.x_k1 222 =", self.x_k1)
        self.P_k = np.array([[1e-8, 1e-8, 1e-8],
                             [1e-8, 1e-8, 1e-8],
                             [1e-8, 1e-8, 1e-8]])

        # 儲存
        self.x_true = np.zeros((3, 1))
        self.k_y = np.zeros((3, 1))
        self.KCP = np.zeros((3, 3))
        self.z = np.zeros((1, 1))
        self.z_data = [self.z.flatten()]
        self.k_y_data = [] # self.k_y.flatten()
        self.x_true_data = []
        # self.x_true_data = [self.x_true.flatten()]
        # print("true_pos_addNoise =", true_pos_addNoise)
        self.x_true_data_noise = [true_pos_addNoise]

        self.KCP_data = [self.KCP.flatten()]
        self.x_k_predict_data = []
        self.P_k_predict_data = []
        self.x_k_update_data = [] # self.x_k1.flatten()
        self.P_k_update_data = [self.P.flatten()]
        self.P_k_data = [self.P_k.flatten()]

        # 存儲估計結果
        self.est_pos = []
        self.est_vel = []
        self.est_acc = []

    def KF(self, measurement):
    # def KF(self, measurement):
        # 預測
        # print("self.A_mat =", self.A_mat)
        # print("self.x_k1 =", self.x_k1)
        self.x = np.dot(self.A_mat, self.x_k1)
        self.x_true_data.append(self.x_k1[0].flatten())
        # print("self.x_k1 =", self.x_k1)
        self.P = np.dot(self.A_mat, np.dot(self.P, self.A_mat.T)) + self.Q
        # print("self.P =", self.P)
        # 更新
        K = np.dot(self.P, self.C.T) / (np.dot(self.C, np.dot(self.P, self.C.T)) + self.R)
        # print("K =", K)
        # print("self.C =", self.C)
        # print("measurement =", measurement)
        # z  = self.C @ measurement
        # print("z =", z)
        # self.z_data.append(z)
        y = measurement - np.dot(self.C, self.x)  # 測量殘差
        # print("y =", y)
        self.k_y = K @ y
        # print("self.k_y =", self.k_y )
        self.k_y_data.append(self.k_y.flatten())
        self.x_k = self.x + np.dot(K, y)
        # print("self.x_k =", self.x_k )
        self.x_k1 = self.x_k
        self.x_k_update_data.append(self.x_k1.flatten())
        # print("self.x =", self.x)
        self.x_k_predict_data.append(self.x.flatten())

        self.KCP = np.dot(np.dot(K, self.C), self.P)
        # print("self.KCP =", self.KCP)
        self.KCP_data.append(self.KCP.flatten())
        self.P_k = np.dot(np.eye(3) - np.dot(K, self.C), self.P)
        # print("self.P_k =", self.P_k)
        self.P = self.P_k
        self.P_k_update_data.append(self.P.flatten())
        self.P_k_predict_data.append(self.P_k.flatten())

        # 儲存估計值
        self.est_pos.append(self.x[0, 0])
        self.est_vel.append(self.x[1, 0])
        self.est_acc.append(self.x[2, 0])

    def get_estimates(self):
        
        # raw data
        est_pos = np.array(self.est_pos)
        est_vel = np.array(self.est_vel)
        est_acc = np.array(self.est_acc)
        # raw_data_all = np.concatenate((est_pos, est_vel, est_acc), axis=1)

        return est_pos, est_vel, est_acc
    
    def getKFData(self):
        # kf x data
        # print("self.x_true_data =", np.array(self.x_true_data).shape)
        # print("self.x_k_update_data =",np.array(self.x_k_update_data).shape)
        self.x_tel = self.x_true_data - np.array(self.x_k_update_data)
        # print(self.k_y_data)
        k_y_data = np.array(self.k_y_data)
        # k_y_data = [np.array(item).reshape(-1) for item in self.k_y_data]
        x_k_update_data = np.array(self.x_k_update_data)
        x_k_update_data = [np.array(item).reshape(-1) for item in x_k_update_data]
        k_y_data = np.array(self.k_y_data)
        x_tel = np.array(self.x_tel)
        # print("x_true_data =", self.x_true_data)
        x_true_data = np.array(self.x_true_data)
        
        x_true_data_noise = np.array(self.x_true_data_noise)
        x_true_data_noise = x_true_data_noise.reshape(total_time*1000, 1)
        # print("x_true_data_noise =", x_true_data_noise.shape)
        x_true_data_noise = [np.array(item).reshape(-1) for item in x_true_data_noise]
        z_data = np.array(self.z_data)
        x_k_predict_data = np.array(self.x_k_predict_data)
        # print("x_true_data =",x_true_data)
        # print("x_true_data_noise =",x_true_data_noise)
        x_data_all = np.concatenate((x_k_update_data, k_y_data, x_tel, x_true_data, x_true_data_noise), axis=1)# me
        # kf P data
        P_k_data = np.array(self.P_k_data)
        P_k_update_data = np.array(self.P_k_update_data)
        P_k_predict_data = np.array(self.P_k_predict_data)
        KCP_data = np.array(self.KCP_data)
        
        P_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)# me

        return x_data_all, P_data_all
        # return x_k_update_data, x_k_predict_data, P_k_update_data, P_k_predict_data, k_y_data, KCP_data, z_data, x_true_data, x_true_data_noise, P_k_data

# 主程式
if __name__ == "__main__":
    # 弦波參數
    A = 5.0          # 振幅
    omega = 2 * np.pi # 角頻率（1 Hz）
    dt = 0.001        # 時間步長
    total_time = 15
    t = np.arange(0, total_time, dt)

    true_pos = A * np.sin(omega * t)      # 真實位置
    true_vel = A * omega * np.cos(omega * t) # 真實速度
    true_acc = -A * omega**2 * np.sin(omega * t) # 真實加速度

    # 添加測量噪聲 true_pos_addNoise
    measurement_noise = 0.1
    true_pos_addNoise = true_pos + np.random.normal(0, measurement_noise, len(true_pos))
    x_true_data = np.array(true_pos)
    x_true_data_noise = np.array(true_pos_addNoise)

    # 系統噪聲協方差矩陣 Q
    # acc CMD variance : 779.2727282720194
    # acc CFD variance : 614691467.3034459
    # acc LAE variance : 4019.7707107859114
    sigma_a2 = 0.1 * (A * omega**2)**2
    Q = 779.2727282720194 * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
                            [dt**3 / 2, dt**2, dt],
                            [dt**2 / 2, dt, 1]]) # acc CMD variance
    Q0 = 400 * np.array([[dt**5/20, dt**4/8, dt**3/6],
                               [dt**4/8, dt**3/3, dt**2/2],
                               [dt**3/6, dt**2/2, dt]]) # acc CMD variance
    Q1 = sigma_a2 * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
                            [dt**3 / 2, dt**2, dt],
                            [dt**2 / 2, dt, 1]])
    Q2 = sigma_a2 * np.array([[dt**5/20, dt**4/8, dt**3/6],
                                [dt**4/8, dt**3/3, dt**2/2],
                                [dt**3/6, dt**2/2, dt]])
    
    Q3 = np.array([[8.54565231e-16, 2.56369441e-12, 5.12738034e-09],
                    [2.56369441e-12, 7.69108152e-09, 1.53821460e-05],
                    [5.12738034e-09, 1.53821460e-05, 3.07642919e-02]]) # CMD + CMD
    Q4 = np.array([[8.15126934e-10, 1.63042644e-06, 9.67863139e-04],
                    [1.63042644e-06, 3.26119887e-03, 1.93573114e+00],
                    [9.67863139e-04, 1.93573114e+00, 3.86731610e+03]]) # CMD + LAE
    Q5 = np.array([[1.47037065e-04, 2.94074131e-01, 4.85786766e+02],
                    [2.94074131e-01, 5.88148263e+02, 9.71573532e+05],
                    [4.85786766e+02, 9.71573532e+05, 1.94314706e+09]]) # CMD + CFD
    
    Q6 = np.array( [[ 2.10953361e-05,  1.54816460e-02, -6.34834622e-06],
                    [ 1.54816460e-02,  3.09716112e+01,  1.28983662e-04],
                    [-6.34834622e-06,  1.28983662e-04,  3.07642919e-02]]) # LSF + CMD
    Q7 = np.array( [[ 1.99376017e-05,  1.43526441e-02, -7.56560364e-02],
                    [ 1.43526441e-02,  2.86501390e+01,  1.67488550e+01],
                    [-7.56560364e-02,  1.67488550e+01,  3.51178648e+03]]) # LSF + LAE
    Q8 = np.array( [[1.19836878e-04, 2.39084171e-01, 4.17566213e+02],
                    [2.39084171e-01, 5.27290908e+02, 8.63098538e+05],
                    [4.17566213e+02, 8.63098538e+05, 1.97316680e+09]]) # LSF + CFD
    
    Q9 = np.array( [[ 4.73435319e-05,  4.56292259e-02, -1.73032293e-06],
                    [ 4.56292259e-02,  9.12584520e+01,  1.60958140e-05],
                    [-1.73032293e-06,  1.60958140e-05,  3.07642919e-02]]) # CFD + CMD
    Q10 = np.array( [[ 5.13857904e-05,  5.25416260e-02, -2.29939423e-01],
                    [ 5.25416260e-02,  1.05295128e+02, -4.55771232e+02],
                    [-2.29939423e-01, -4.55771232e+02,  3.97185147e+03]]) # CFD + LAE
    Q11 = np.array( [[2.04064838e-04, 3.09607639e-01, 6.19476684e+02],
                    [3.09607639e-01, 5.21419926e+02, 1.04251985e+06],
                    [6.19476684e+02, 1.04251985e+06, 2.08717058e+09]]) # CFD + CFD

    # 測量噪聲方差
    R = np.array([[measurement_noise**2]])
    # R = np.array([[measurement_noise]])

    # 初始化卡爾曼濾波器
    for i in range(len(true_pos_addNoise)):
        kf = KalmanFilter(dt=dt, Q=Q, R=R) # 
        kf0 = KalmanFilter(dt=dt, Q=Q0, R=R)
        kf1 = KalmanFilter(dt=dt, Q=Q1, R=R)
        kf2 = KalmanFilter(dt=dt, Q=Q2, R=R)
        kf3 = KalmanFilter(dt=dt, Q=Q3, R=R)
        kf4 = KalmanFilter(dt=dt, Q=Q4, R=R)
        kf5 = KalmanFilter(dt=dt, Q=Q5, R=R)
        kf6 = KalmanFilter(dt=dt, Q=Q6, R=R)
        kf7 = KalmanFilter(dt=dt, Q=Q7, R=R)
        kf8 = KalmanFilter(dt=dt, Q=Q8, R=R)
        kf9 = KalmanFilter(dt=dt, Q=Q9, R=R)
        kf10 = KalmanFilter(dt=dt, Q=Q10, R=R)
        kf11 = KalmanFilter(dt=dt, Q=Q11, R=R)

    # 執行濾波
    for z in true_pos_addNoise:
        kf.KF(z)
        kf0.KF(z)
        kf1.KF(z)
        kf2.KF(z)
        kf3.KF(z)
        kf4.KF(z)
        kf5.KF(z)
        kf6.KF(z)
        kf7.KF(z)
        kf8.KF(z)
        kf9.KF(z)
        kf10.KF(z)
        kf11.KF(z)

    # 獲取估計結果
    est_pos, est_vel, est_acc = kf.get_estimates()
    est_pos0, est_vel0, est_acc0 = kf0.get_estimates()
    est_pos1, est_vel1, est_acc1 = kf1.get_estimates()
    est_pos2, est_vel2, est_acc2 = kf2.get_estimates()
    est_pos3, est_vel3, est_acc3 = kf3.get_estimates()
    est_pos4, est_vel4, est_acc4 = kf4.get_estimates()
    est_pos5, est_vel5, est_acc5 = kf5.get_estimates()
    est_pos6, est_vel6, est_acc6 = kf6.get_estimates()
    est_pos7, est_vel7, est_acc7 = kf7.get_estimates()
    est_pos8, est_vel8, est_acc8 = kf8.get_estimates()
    est_pos9, est_vel9, est_acc9 = kf9.get_estimates()
    est_pos10, est_vel10, est_acc10 = kf10.get_estimates()
    est_pos11, est_vel11, est_acc11 = kf11.get_estimates()

#---------------------------------------輸出數據.txt-------------------------------------------#
    # 決定以哪個Q值得數據進行儲存輸出
    x_data_all, P_data_all = kf2.getKFData()
    np.savetxt('./sim_data/dataset/x_data_all.txt', x_data_all, delimiter=' ')
    np.savetxt('./sim_data/dataset/P_data_all.txt', P_data_all, delimiter=' ')
    # raw data save
    # est_pos = [np.array(item).reshape(-1) for item in est_pos]
    # est_vel = [np.array(item).reshape(-1) for item in est_vel]
    # est_acc = [np.array(item).reshape(-1) for item in est_acc]
    true_pos_shape = true_pos.reshape(-1, 1) 
    true_vel_shape = true_vel.reshape(-1, 1)
    true_acc_shape = true_acc.reshape(-1, 1)
    raw_data_all = np.concatenate((true_pos_shape, true_vel_shape, true_acc_shape), axis=1)
    np.savetxt('./sim_data/dataset/raw_data_all.txt', raw_data_all, delimiter=' ')

#---------------------------------------計算covariance-------------------------------------------#
    # 計算covariance
    Pos_CFD_est, Vel_CFD_est, Acc_CFD_est = CFD.CFD(true_pos_addNoise) 
    Vel_LSF_est = LSF.LSF14(true_pos_addNoise)
    Acc_LAE_est = LAE.LAE(true_pos_addNoise, 0.001)

    zk = []
    err = []

    for i in range(len(true_pos_addNoise)-1):
        dt = 0.001
        zk = np.array([[true_pos[i]], 
                        [Vel_CFD_est[i]],
                        [Acc_CFD_est[i]]]) # vel為LSF1/4
        zk_1 = np.array([[true_pos[i+1]], 
                        [Vel_CFD_est[i+1]],
                        [Acc_CFD_est[i+1]]]) # vel為LSF1/4
        A = np.array([[1, dt, 0.5*dt**2],
                    [0, 1, dt],
                    [0, 0, 1 ]])
        error = zk_1 - A @ zk
        err.append(error)
    err = np.array(err).squeeze()

    print('-----------------------------------------------------------------')
    print("np.cov(err) = \n", np.cov(err.T)) 
    # variance_p = np.var(err[:, 0])
    # print("variance_p = ", variance_p)
    # variance_v = np.var(err[:, 1])
    # print("variance_v = ", variance_v)
    # variance_a = np.var(err[:, 2])
    # print("variance_a = ", variance_a)
    print('-----------------------------------------------------------------')
    print("acc estimate")
    print("acc CMD variance :", np.var(true_acc))
    print("acc CFD variance :", np.var(Acc_CFD_est))
    print("acc LAE variance :", np.var(Acc_LAE_est))
    print('-----------------------------------------------------------------')

#---------------------------------------可視化結果-------------------------------------------#
    # 可視化結果
    #-------------------------------------不同模型假設比較-------------------------------------------#
    # pos
    
    plt.figure(figsize=(8, 6))
    plt.plot(true_pos, "black", label="True Pos", linewidth=1)
    plt.plot(true_vel, "blue", label="True Pos", linewidth=1)
    plt.plot(true_acc, "red", label="True Pos", linewidth=1)
    plt.legend()
    plt.title("ground truth", loc="center")

    # print("est_pos =", est_pos)
    plt.figure(figsize=(8, 6))
    # plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, est_pos, label="KF Est Pos by Q", linewidth=1)
    plt.plot(t, est_pos0, label="KF Est Pos by Q0", linewidth=1)
    plt.plot(t, est_pos1, label="KF Est Pos by Q1", linewidth=1)
    plt.plot(t, est_pos2, label="KF Est Pos by Q2", linewidth=1)
    plt.legend()
    plt.title("KF with different model assumption", loc="center")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    plt.plot(t, est_vel, label="KF Est Vel by with Q", linewidth=1)
    plt.plot(t, est_vel0, label="KF Est Vel by with Q0", linewidth=1)
    plt.plot(t, est_vel1, label="KF Est Vel by with Q1", linewidth=1)
    plt.plot(t, est_vel2, label="KF Est Vel by with Q2", linewidth=1)
    # plt.legend()

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    plt.plot(t, est_acc, label="KF Est Acctby ion with Q", linewidth=1)
    plt.plot(t, est_acc0, label="KF Est Acctby ion with Q0", linewidth=1)
    plt.plot(t, est_acc1, label="KF Est Acctby ion with Q1", linewidth=1)
    plt.plot(t, est_acc2, label="KF Est Acctby ion with Q2", linewidth=1)
    # plt.legend()
    plt.tight_layout()

    #-------------------------------------不同估測方法計算Q比較 vel:CMD-------------------------------------------#
    # pos
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, est_pos3, label="KF Est Pos by Q3 ", linewidth=1)
    plt.plot(t, est_pos4, label="KF Est Pos by Q4", linewidth=1)
    plt.plot(t, est_pos5, label="KF Est Pos by Q5", linewidth=1)
    plt.legend()
    plt.title("KF with different estimator calculate Q -vel:CMD")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    plt.plot(t, est_vel3, label="KF Est Vel by with Q3", linewidth=1)
    plt.plot(t, est_vel4, label="KF Est Vel by with Q4", linewidth=1)
    plt.plot(t, est_vel5, label="KF Est Vel by with Q5", linewidth=1)
    # plt.legend()

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    plt.plot(t, est_acc3, label="KF Est Acctby ion with Q3", linewidth=1)
    plt.plot(t, est_acc4, label="KF Est Acctby ion with Q4", linewidth=1)
    plt.plot(t, est_acc5, label="KF Est Acctby ion with Q5", linewidth=1)
    # plt.legend()
    plt.tight_layout()

    #-------------------------------------不同估測方法計算Q比較 vel:LSF-------------------------------------------#
    # pos
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    plt.plot(t, est_pos6, label="KF Est Pos by Q6", linewidth=1)
    plt.plot(t, est_pos7, label="KF Est Pos by Q7", linewidth=1)
    plt.plot(t, est_pos8, label="KF Est Pos by Q8", linewidth=1)
    plt.legend()
    plt.title("KF with different estimator calculate Q -vel:LSF")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    plt.plot(t, est_vel6, label="KF Est Vel by with Q6", linewidth=1)
    plt.plot(t, est_vel7, label="KF Est Vel by with Q7", linewidth=1)
    plt.plot(t, est_vel8, label="KF Est Vel by with Q8", linewidth=1)
    # plt.legend()

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    plt.plot(t, est_acc6, label="KF Est Acctby ion with Q6", linewidth=1)
    plt.plot(t, est_acc7, label="KF Est Acctby ion with Q7", linewidth=1)
    plt.plot(t, est_acc8, label="KF Est Acctby ion with Q8", linewidth=1)
    # plt.legend()
    plt.tight_layout()

    #-------------------------------------不同估測方法計算Q比較 vel:CFD-------------------------------------------#
    # pos
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    # plt.plot(t, est_pos1, label="KF Est Pos by Q1", linewidth=1)
    # plt.plot(t, est_pos2, label="KF Est Pos by Q2", linewidth=1)
    # plt.plot(t, est_pos3, label="KF Est Pos by Q3 ", linewidth=1)
    # plt.plot(t, est_pos4, label="KF Est Pos by Q4", linewidth=1)
    # plt.plot(t, est_pos5, label="KF Est Pos by Q5", linewidth=1)
    # plt.plot(t, est_pos6, label="KF Est Pos by Q6", linewidth=1)
    # plt.plot(t, est_pos7, label="KF Est Pos by Q7", linewidth=1)
    # plt.plot(t, est_pos8, label="KF Est Pos by Q8", linewidth=1)
    plt.plot(t, est_pos9, label="KF Est Pos by Q9", linewidth=1)
    plt.plot(t, est_pos10, label="KF Est Pos by Q10", linewidth=1)
    plt.plot(t, est_pos11, label="KF Est Pos by Q11", linewidth=1)
    plt.legend()
    plt.title("KF with different estimator calculate Q -vel:CFD")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    # plt.plot(t, est_vel1, label="KF Est Vel by with Q1", linewidth=1)
    # plt.plot(t, est_vel2, label="KF Est Vel by with Q2", linewidth=1)
    # plt.plot(t, est_vel3, label="KF Est Vel by with Q3", linewidth=1)
    # plt.plot(t, est_vel4, label="KF Est Vel by with Q4", linewidth=1)
    # plt.plot(t, est_vel5, label="KF Est Vel by with Q5", linewidth=1)
    # plt.plot(t, est_vel6, label="KF Est Vel by with Q6", linewidth=1)
    # plt.plot(t, est_vel7, label="KF Est Vel by with Q7", linewidth=1)
    # plt.plot(t, est_vel8, label="KF Est Vel by with Q8", linewidth=1)
    plt.plot(t, est_vel9, label="KF Est Vel by with Q9", linewidth=1)
    plt.plot(t, est_vel10, label="KF Est Vel by with Q10", linewidth=1)
    plt.plot(t, est_vel11, label="KF Est Vel by with Q11", linewidth=1)
    # plt.legend()

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    # plt.plot(t, est_acc1, label="KF Est Acctby ion with Q1", linewidth=1)
    # plt.plot(t, est_acc2, label="KF Est Acctby ion with Q2", linewidth=1)
    # plt.plot(t, est_acc3, label="KF Est Acctby ion with Q3", linewidth=1)
    # plt.plot(t, est_acc4, label="KF Est Acctby ion with Q4", linewidth=1)
    # plt.plot(t, est_acc5, label="KF Est Acctby ion with Q5", linewidth=1)
    # plt.plot(t, est_acc6, label="KF Est Acctby ion with Q6", linewidth=1)
    # plt.plot(t, est_acc7, label="KF Est Acctby ion with Q7", linewidth=1)
    # plt.plot(t, est_acc8, label="KF Est Acctby ion with Q8", linewidth=1)
    plt.plot(t, est_acc9, label="KF Est Acctby ion with Q9", linewidth=1)
    plt.plot(t, est_acc10, label="KF Est Acctby ion with Q10", linewidth=1)
    plt.plot(t, est_acc11, label="KF Est Acctby ion with Q11", linewidth=1)
    # plt.legend()
    plt.tight_layout()

    #-------------------------------------不同Q估測結果比較 0、2、4、7、10-------------------------------------------#
    # pos
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, true_pos, "black", label="True Pos", linewidth=1)
    plt.plot(t, true_pos_addNoise, "blue", label="True Pos addNoise", linestyle="dotted")
    # plt.plot(t, est_pos, label="KF Est Pos by Q", linewidth=1)
    plt.plot(t, est_pos0, label="KF Est Pos by Q0", linewidth=1)
    # plt.plot(t, est_pos1, label="KF Est Pos by Q1", linewidth=1)
    plt.plot(t, est_pos2, label="KF Est Pos by Q2", linewidth=1)
    # plt.plot(t, est_pos3, label="KF Est Pos by Q3 ", linewidth=1)
    plt.plot(t, est_pos4, label="KF Est Pos by Q4", linewidth=1)
    # plt.plot(t, est_pos5, label="KF Est Pos by Q5", linewidth=1)
    # plt.plot(t, est_pos6, label="KF Est Pos by Q6", linewidth=1)
    plt.plot(t, est_pos7, label="KF Est Pos by Q7", linewidth=1)
    # plt.plot(t, est_pos8, label="KF Est Pos by Q8", linewidth=1)
    # plt.plot(t, est_pos9, label="KF Est Pos by Q9", linewidth=1)
    plt.plot(t, est_pos10, label="KF Est Pos by Q10", linewidth=1)
    # plt.plot(t, est_pos11, label="KF Est Pos by Q11", linewidth=1)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.title("KF with different Q estimate result comparison")

    # vel
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 2)
    plt.plot(t, true_vel, "black", label="True Velocity", linewidth=1)
    # plt.plot(t, est_vel, label="KF Est Vel by with Q", linewidth=1)
    plt.plot(t, est_vel0, label="KF Est Vel by with Q0", linewidth=1)
    # plt.plot(t, est_vel1, label="KF Est Vel by with Q1", linewidth=1)
    plt.plot(t, est_vel2, label="KF Est Vel by with Q2", linewidth=1)
    # plt.plot(t, est_vel3, label="KF Est Vel by with Q3", linewidth=1)
    plt.plot(t, est_vel4, label="KF Est Vel by with Q4", linewidth=1)
    # plt.plot(t, est_vel5, label="KF Est Vel by with Q5", linewidth=1)
    # plt.plot(t, est_vel6, label="KF Est Vel by with Q6", linewidth=1)
    plt.plot(t, est_vel7, label="KF Est Vel by with Q7", linewidth=1)
    # plt.plot(t, est_vel8, label="KF Est Vel by with Q8", linewidth=1)
    # plt.plot(t, est_vel9, label="KF Est Vel by with Q9", linewidth=1)
    plt.plot(t, est_vel10, label="KF Est Vel by with Q10", linewidth=1)
    # plt.plot(t, est_vel11, label="KF Est Vel by with Q11", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("vel")

    # acc
    # plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 3)
    plt.plot(t, true_acc, "black", label="True Acceleration", linewidth=1)
    # plt.plot(t, est_acc, label="KF Est Acctby ion with Q", linewidth=1)
    plt.plot(t, est_acc0, label="KF Est Acctby ion with Q0", linewidth=1)
    # plt.plot(t, est_acc1, label="KF Est Acctby ion with Q1", linewidth=1)
    plt.plot(t, est_acc2, label="KF Est Acctby ion with Q2", linewidth=1)
    # plt.plot(t, est_acc3, label="KF Est Acctby ion with Q3", linewidth=1)
    # plt.plot(t, est_acc4, label="KF Est Acctby ion with Q4", linewidth=1)
    # plt.plot(t, est_acc5, label="KF Est Acctby ion with Q5", linewidth=1)
    # plt.plot(t, est_acc6, label="KF Est Acctby ion with Q6", linewidth=1)
    # plt.plot(t, est_acc7, label="KF Est Acctby ion with Q7", linewidth=1)
    # plt.plot(t, est_acc8, label="KF Est Acctby ion with Q8", linewidth=1)
    # plt.plot(t, est_acc9, label="KF Est Acctby ion with Q9", linewidth=1)
    # plt.plot(t, est_acc10, label="KF Est Acctby ion with Q10", linewidth=1)
    # plt.plot(t, est_acc11, label="KF Est Acctby ion with Q11", linewidth=1)
    # plt.legend()
    plt.xlabel("t")
    plt.ylabel("acc")
    plt.tight_layout()
    


    plt.show()
