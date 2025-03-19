import numpy as np
import motor.ImportData as ImportData, motor.Cal as Cal
import KF, LSTM, AddNoise

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P, u):
        self.A = A  # 狀態轉移矩陣
        self.B = B  # 控制矩陣
        self.H = H  # 觀測矩陣
        self.Q = Q  # 過程噪聲
        self.R = R  # 測量噪聲
        self.P = P  # 誤差協方差矩陣
        # self.z = z  # 實際量測值
        self.u = u
        self.x = np.zeros((A.shape[1], 1))  # 初始狀態
    
    def predict(self):
        # 狀態預測
        self.x = self.A @ self.x + self.B @ self.u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x
    
    def update(self, z): # z為輸入PosCmd
        # 卡爾曼增益
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        # 狀態更新
        y_tel = (z - self.H @ self.x)
        k_y = K @ y_tel
        self.x = self.x +  k_y 
        # 誤差協方差更新
        KCP = K @ self.H @ self.P
        self.P = np.eye(self.A.shape[0]) @ self.P - K @ self.H @ self.P
        # self.P = (np.eye(self.A.shape[0]) - K @ self.H) @ self.P
        return self.x, self.P, K, k_y, KCP

def KF_Process(path1, path2):
    # 相關參數
    SamplingTime = 0.001
    CPI = 1600
    r = 11.6287
    Motordata, Mousedata = ImportData.ImportData(path1, path2)
    MouseTime, MotorTime, mouseX, mouseY, Pos, PosCmd, Vel, VelCmd, AccCmd, TorCtrl, mousedata_data, mouse_displacement, mouse_real_Pos = Cal.Cal(Mousedata, Motordata, SamplingTime, CPI) 
    PosCmd_AddNoise, VelCmd_AddNoise, AccCmd_AddNoise, noice_percent_record = AddNoise.AddNoice(PosCmd, VelCmd, AccCmd)
    t = np.arange(0, (len(Motordata[:,0])) * SamplingTime, SamplingTime)

    # 初始化卡爾曼濾波器參數
    dt = 0.001
    # 狀態轉移矩陣
    A = np.array([[1, dt, 0.5*dt**2],
                [0, 1, dt],
                [0, 0, 1 ]])
    # 控制矩陣
    B = np.array([[0.5*dt**2],
                [dt],
                [1]])
    H = np.array([[1, 0, 0]]) # 1/cpi 
    Q = np.array([[1e-6, 0, 0],
                  [0, 0.05501, 0],
                  [0, 0, 5501*289*10**-3]]) 
    # 過程噪聲
    # Q = np.array([[ 4.03005280e-06, 0, 0],
    #                 [0,  1.73656793e+01,  0],
    #                 [0,  0,  6.67572647e+05]]) 
    # 測量噪聲
    R = 0.00126*2 #與誤差有關 -> 影響平滑度
    # 誤差協方差矩陣
    # P = np.array([[1e-4, 0, 0],
    #                 [0, 1e-4, 0],
    #                 [0, 0, 1e-4]])
    P = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])
    z = Pos
    # 控制輸入
    u = np.zeros((1, 1))

    # 創建卡爾曼濾波器實例
    KF = KalmanFilter(A, B, H, Q, R, P, u)
    # Import data:trainind & validation
    x_kf_update_data = []
    P_kf_update_data = []
    K_update_data = []
    k_y_update_data = []
    KCP_data = []
    for k in range(len(z)):
        # 卡爾曼濾波器預測步驟
        x_pred = KF.predict()
        
        # 創新量
        y_k = z[k] - H @ x_pred

        # 卡爾曼濾波器更新步驟
        x_kf_update, P_kf_update, K_update, k_y_update, KCP = KF.update(z[k])
        # DataSet
        x_kf_update_data.append(x_kf_update.flatten())  # 存儲每次的狀態更新
        P_kf_update_data.append(P_kf_update.flatten())  # 存儲每次的誤差協方差矩陣
        K_update_data.append(K_update.flatten())
        k_y_update_data.append(k_y_update.flatten())
        KCP_data.append(KCP.flatten())
    return x_kf_update_data, P_kf_update_data, K_update_data, k_y_update_data, KCP_data, H, Pos, PosCmd, VelCmd, AccCmd, PosCmd_AddNoise, VelCmd_AddNoise, AccCmd_AddNoise