import numpy as np
import ImportData, Cal

a = 0
def KF(dt, pos):
    A = np.array([[1, dt, 0.5*dt**2],
                  [0, 1, dt],
                  [0, 0, 1 ]])
    B = np.array([[0.5*dt**2],
                  [dt],
                  [1]])
    u = AccCmd
    C = np.array([[1, 0, 0]]) # 1/cpi 
    Q = np.array([[1e-6, 0, 0],
                  [0, 0.05501, 0],
                  [0, 0, 5501*289*10**-3]]) 
    # Q = np.array([[ 4.03005280e-06, 0, 0],
    #               [0,  1.73656793e+01,  0],
    #               [0,  0,  6.67572647e+05]]) 
    R = 0.00126*2 # 3*10e-4 # 3000 # 5000 # 5000 #150 #500 #1000 #100 #10 #1.5 #1 #與誤差有關 -> 影響平滑度
    P = np.array([[1e-4, 0, 0],
                  [0, 1e-4, 0],
                  [0, 0, 1e-4]])
    Wt = 0
    pose = np.zeros(len(pos))
    vele = np.zeros(len(pos))
    acce = np.zeros(len(pos))
    xm = np.zeros((3, 1))  
    Pm = P
    km_y_data = []
    for i in range(len(pos)): # m = measurement;p = predict
        Pp = np.dot(np.dot(A, Pm), A.T) + Q
        xp = np.dot(A, xm) + Wt
        # xp = np.dot(A, xm) + 1e0 * np.dot(B, u[i]) + Wt 
        Km = np.dot(Pp, C.T) / (np.dot(np.dot(C, Pp), C.T) + R)
        y = (pos[i] - np.dot(C, xp))
        km_y = np.dot(Km, y)
        xm = xp + km_y
        Pm = np.dot((np.eye(3) - np.dot(Km, C)), Pp)
        pose[i] = xm[0, 0]
        vele[i] = xm[1, 0]
        acce[i] = xm[2, 0]
        km_y_data.append(km_y)
    return pose, vele, acce

if __name__ == "__main__":
    # Constant
    SamplingTime = 0.001
    ## 木盤半徑12.5 壓克力盤半徑12.53
    wood = 12.5
    plastic = 11.945 #11.62 #11.14 # 12.53
    
    ## 量測半徑
    r = 11.6287
    ## 讀取檔案
    path1 = ['motor_dataset/IPS650_G50_motion.txt'] #馬達資料.txt路徑
    path2 = ['motor_dataset/IPS650_G50_mouse.txt']  #滑鼠資料.txt路徑
    
    Motordata, Mousedata = ImportData.ImportData(path1, path2)
    Pos, PosCmd, Vel, VelCmd, AccCmd, TorCtrl = Cal.Cal(Motordata, SamplingTime) 
    t = np.arange(0, (len(Motordata[:,0])) * SamplingTime, SamplingTime)
    pose, vele, acce = KF(SamplingTime, Pos)
    Motor_true_data = []
    PosCmd = np.expand_dims(PosCmd, axis=1)  # 將一維數組擴展為 n x 1
    VelCmd = np.expand_dims(VelCmd, axis=1)
    AccCmd = np.expand_dims(AccCmd, axis=1)
    pose = np.expand_dims(pose, axis=1)
    vele = np.expand_dims(vele, axis=1)
    acce = np.expand_dims(acce, axis=1)
    Motor_true_data = np.concatenate((PosCmd, VelCmd, AccCmd, pose, vele, acce), axis=1)
    
    # x_data_all_np = x_data_all.get()
    # P_data_all_np = P_data_all.get()
    # 使用 numpy.savetxt 将其保存到 txt 文件中
    np.savetxt('Motor_true_data.txt', Motor_true_data, delimiter=' ')

