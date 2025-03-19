import numpy as np

a = 0
def KF(dt, pos, AccCmd):
    A = np.array([[1, dt, 0.5*dt**2],
                  [0, 1, dt],
                  [0, 0, 1 ]])
    B = np.array([[0.5*dt**2],
                  [dt],
                  [1]])
    u = AccCmd
    C = np.array([[1, 0, 0]]) # 1/cpi 
    # Q = np.array([[1e-6, 0, 0],
    #               [0, 0.05501, 0],
    #               [0, 0, 5501*289*10**-3]]) 
    Q = np.array([[ 4.03005280e-06, 0, 0],
                  [0,  1.73656793e+01,  0],
                  [0,  0,  6.67572647e+05]]) 
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
    for i in range(len(pos)): # m = measurement;p = predict
        Pp = np.dot(np.dot(A, Pm), A.T) + Q
        xp = np.dot(A, xm) + Wt
        # xp = np.dot(A, xm) + 1e0 * np.dot(B, u[i]) + Wt 
        Km = np.dot(Pp, C.T) / (np.dot(np.dot(C, Pp), C.T) + R)
        y = (pos[i] - np.dot(C, xp))
        xm = xp + np.dot(Km, y)
        Pm = np.dot((np.eye(3) - np.dot(Km, C)), Pp)
        pose[i] = xm[0, 0]
        vele[i] = xm[1, 0]
        acce[i] = xm[2, 0]
    return pose, vele, acce