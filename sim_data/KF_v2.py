import numpy as np

a = 0
def KF(dt, pos, SIGMA):
    A = np.array([[1, dt, 0.5*dt**2],
                  [0, 1, dt],
                  [0, 0, 1 ]])
    # A = np.array([[1, dt, 0.5*dt**2],
    #               [0.05, 1, dt],
    #               [-0.05**2, 0, 0 ]])
    B = np.array([[0.5*dt**2],
                  [dt],
                  [1]])
    # u = AccCmd
    C = np.array([[1, 0, 0]]) # 1/cpi 
    # Q = np.array([[1e-6, 0, 0],
    #               [0, 0.05501, 0],
    #               [0, 0, 5501*289*10**-3]]) 
    # Q = np.array([[4.03005280e-06, 0, 0],
    #               [0,  1.73656793e+01,  0],
    #               [0,  0,  6.67572647e+05]])
    
    # 弦波-等加速度模型假設
    k = 1e-1
    sigma = k * (2 * 0.05**2)**2
    # Q = sigma * np.array([[dt**5/20, dt**4/8, dt**3/6],
    #                         [dt**4/8, dt**3/3, dt**2/2],
    #                         [dt**3/6, dt**2/2, dt]])
    Q = sigma * np.array([[dt**4/4, dt**3/2, dt**2/2],
                            [dt**3/2, dt**2, dt],
                            [dt**2/2, dt, 1]]) 
    
    # 等加速度模型假設
    # Q = SIGMA * np.array([[dt**5/20, dt**4/8, dt**3/6],
    #                        [dt**4/8, dt**3/3, dt**2/2],
    #                        [dt**3/6, dt**2/2, dt]]) 
    # Q = 0.001223683523833341 * np.array([[dt**5/20, dt**4/8, dt**3/6],
    #                        [dt**4/8, dt**3/3, dt**2/2],
    #                        [dt**3/6, dt**2/2, dt]]) # CFD
    # Q = 35.05406011659647 * np.array([[dt**5/20, dt**4/8, dt**3/6],
    #                        [dt**4/8, dt**3/3, dt**2/2],
    #                        [dt**3/6, dt**2/2, dt]]) # LAE
    # Q = 0.001223683523833341 * np.array([[dt**5/20, dt**4/8, dt**3/6],
    #                        [dt**4/8, dt**3/3, dt**2/2],
    #                        [dt**3/6, dt**2/2, dt]]) # CMD
    
    # 變加速度模型假設
    # Q = SIGMA * np.array([[dt**4/4, dt**3/2, dt**2/2],
    #                                   [dt**3/2, dt**2, dt],
    #                                   [dt**2/2, dt, 1]]) 
    # Q = 0.001223683523833341 * np.array([[dt**4/4, dt**3/2, dt**2/2],
    #                                   [dt**3/2, dt**2, dt],
    #                                   [dt**2/2, dt, 1]]) # CFD
    # Q = 3625.97 * np.array([[dt**4/4, dt**3/2, dt**2/2],
    #                        [dt**3/2, dt**2, dt],
    #                        [dt**2/2, dt, 1]]) # LAE
    # Q = 35.05406011659647 * np.array([[dt**4/4, dt**3/2, dt**2/2],
    #                        [dt**3/2, dt**2, dt],
    #                        [dt**2/2, dt, 1]]) # LAE
    # Q = 0.001223683523833341 * np.array([[dt**4/4, dt**3/2, dt**2/2],
    #                        [dt**3/2, dt**2, dt],
    #                        [dt**2/2, dt, 1]]) # CMD
    # Q = np.array([[4.08063224e-27, 3.52234601e-14, 7.04550127e-11],
    #                 [3.52234601e-14, 1.10921637e+04, 2.21401106e+07],
    #                 [7.04550127e-11, 2.21401106e+07, 4.42802211e+10]]) # MOUSE CFD


    # R = 0.00126*2 # 3*10e-4 # 3000 # 5000 # 5000 #150 #500 #1000 #100 #10 #1.5 #1 #與誤差有關 -> 影響平滑度
    # R = 0.0022151714012120923 # POS-POSCMD的變異數
    R = 0.001 # sim data
    P = np.array([[1e-4, 0, 0],
                  [0, 1e-4, 0],
                  [0, 0, 1e-4]])
    # P = np.array([[ 4.03005280e-06, -3.83997884e-04, -5.90703882e+00],
    #               [-3.83997884e-04,  1.73656793e+01,  2.89105567e+04],
    #               [-5.90703882e+00,  2.89105567e+04,  6.67572647e+07]])
    Wt = 0
    pose = np.zeros(len(pos))
    vele = np.zeros(len(pos))
    acce = np.zeros(len(pos))
    xm = np.zeros((3, 1))  
    Pm = P
    for i in range(len(pos)): # m = measurement;p = predict
        Pp = np.dot(np.dot(A, Pm), A.T) + Q
        xp = np.dot(A, xm) + Wt
        # Bu = np.dot(B, u[i])
        # print('Bu =', Bu)
        # xp = np.dot(A, xm) + Bu + Wt 
        Km = np.dot(Pp, C.T) / (np.dot(np.dot(C, Pp), C.T) + R)
        y = (pos[i] - np.dot(C, xp))
        xm = xp + np.dot(Km, y)
        Pm = np.dot((np.eye(3) - np.dot(Km, C)), Pp)
        pose[i] = xm[0, 0]
        vele[i] = xm[1, 0]
        acce[i] = xm[2, 0]
    return pose, vele, acce

# ORIGINAL
def KF_ORIG(dt, pos):
    # A = np.array([[1, dt, 0.5*dt**2],
    #               [0, 1, dt],
    #               [0, 0, 1 ]])
    A = np.array([[1, dt, 0.5*dt**2],
                  [0.05, 1, dt],
                  [-0.05**2, 0, 0 ]])
    B = np.array([[0.5*dt**2],
                  [dt],
                  [1]])
    # u = AccCmd
    C = np.array([[1, 0, 0]]) # 1/cpi 
    # Q = np.array([[1e-6, 0, 0],
    #               [0, 0.05501, 0],
    #               [0, 0, 5501*289*10**-3]]) 
    Q = np.array( [[ 7.09368306e-04,  4.73461788e-01, -3.80660676e-01],
                    [ 4.73461788e-01,  8.71775352e+02,  1.02284107e+03],
                    [-3.80660676e-01,  1.02284107e+03,  1.44587281e+05]])
    # Q = np.array( [[ 0.01,  0.01, 0.01],
    #                 [ 0.01,  0.01,  0.01],
    #                 [0.01,  0.01,  0.01]])
    # Q = np.array([[ 1.02835843e-07, 0, 0],
    #                 [0,  9.91823076e-08,  0],
    #                 [0,  0,  1.26962432e-07]])
    # Q = np.array([[4.03005280e-06, 0, 0],
    #               [0,  1.73656793e+01,  0],
    #               [0,  0,  6.67572647e+05]])
    # Q = np.array([[1.14494042e-05, 3.07764544e-03, 1.62770948e-02],
    #               [3.07764544e-03, 1.53484685e+00, 1.42351785e+01],
    #               [1.62770948e-02, 1.42351785e+01, 1.04866619e+03]]) # LAE
    # Q = np.array([[ 4.03005280e-06, -3.83997884e-04, -5.90703882e+00],
    #               [-3.83997884e-04,  1.73656793e+01,  2.89105567e+04],
    #               [-5.90703882e+00,  2.89105567e+04,  6.67572647e+07]]) # CFD
    # Q = np.array([[1.19728937e-07, 2.04532016e-06, 2.04531994e-03],
    #               [2.04532016e-06, 2.71624020e-03, 2.71624015e+00],
    #               [2.04531994e-03, 2.71624015e+00, 2.71624010e+03]]) # CMD
    # Q = np.array([[4.08063224e-27, 3.52234601e-14, 7.04550127e-11],
    #                 [3.52234601e-14, 1.10921637e+04, 2.21401106e+07],
    #                 [7.04550127e-11, 2.21401106e+07, 4.42802211e+10]]) # MOUSE CFD
    # Q = np.array([[ 3.57026540e-03,  1.60612883e+00, -1.47234703e+01],
    #                 [ 1.60612883e+00,  3.23561743e+03, -1.26937496e+04],
    #                 [-1.47234703e+01, -1.26937496e+04,  1.59845381e+05]]) # MOUSE LAE

    # Q RESEARCH
    # Q = np.array([[  2.10106472e-02, -3.83997884e-04, -5.90703882e+00],
    #                 [-3.83997884e-04,  2.50125783e-02,  2.89105567e+04],
    #                 [-5.90703882e+00,  2.89105567e+04,  5.26239967e+05]])
    # Q = np.array([[1.80478348e-02, 0, 0],
    #               [0,  2.35069343e-02,  0],
    #               [0,  0,  5.64204517e+05]])


    # R = 0.00126*2 # 3*10e-4 # 3000 # 5000 # 5000 #150 #500 #1000 #100 #10 #1.5 #1 #與誤差有關 -> 影響平滑度
    # R = 0.0022151714012120923 # POS-POSCMD的變異數
    R = 0.1 # sim data
    P = np.array([[1e-4, 0, 0],
                  [0, 1e-4, 0],
                  [0, 0, 1e-4]])
    Wt = 0
    pose = np.zeros(len(pos))
    vele = np.zeros(len(pos))
    acce = np.zeros(len(pos))
    xm = np.zeros((3, 1))  
    Pm = P
    for i in range(len(pos)): # m = measurement;p = predict
        Pp = np.dot(np.dot(A, Pm), A.T) + Q
        xp = np.dot(A, xm) + Wt
        # Bu = np.dot(B, u[i])
        # print('Bu =', Bu)
        # xp = np.dot(A, xm) + Bu + Wt 
        Km = np.dot(Pp, C.T) / (np.dot(np.dot(C, Pp), C.T) + R)
        y = (pos[i] - np.dot(C, xp))
        xm = xp + np.dot(Km, y)
        Pm = np.dot((np.eye(3) - np.dot(Km, C)), Pp)
        pose[i] = xm[0, 0]
        vele[i] = xm[1, 0]
        acce[i] = xm[2, 0]
    return pose, vele, acce