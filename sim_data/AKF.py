import numpy as np
import matplotlib.pyplot as plt

## 速度\加速度Q值分開調整
def AKF_2(dt, Pos, PosCmd, VelCmd, AccCmd):
    A = np.array([[1, dt, 0.5*dt**2],
                  [0, 1, dt],
                  [0, 0, 1 ]])
    B = np.array([[0.5*dt**2],
                  [dt],
                  [1]])
    u = AccCmd
    C = np.array([[1, 0, 0]]) 
    
    # Q = np.array([[0.0, 0, 0],
    #               [0, 0.0, 0],
    #               [0, 0, 0.0]]) 
    c = 5.0          # 振幅
    omega = 2 * np.pi # 角頻率（1 Hz）
    sigma_a2 = 0.1 * (c * omega**2)**2
    Q = sigma_a2 * np.array([[dt**5/20, dt**4/8, dt**3/6],
                                [dt**4/8, dt**3/3, dt**2/2],
                                [dt**3/6, dt**2/2, dt]])
    # Q = np.array([[0.1, 0, 0],
    #               [0, 0.1, 0],
    #               [0, 0, 0.1]]) 
    # Q = np.array([[1e-6, 0, 0],
    #               [0, 0.05501, 0],
    #               [0, 0, 5501*289*10**-3]]) 
    #----------------------------------------------------------------#
    # USING CFD EST ACC
    # np.cov(err) =  [[ 4.03005280e-06 -3.83997884e-04 -5.90703882e+00]
    #                 [-3.83997884e-04  1.73656793e+01  2.89105567e+04]
    #                 [-5.90703882e+00  2.89105567e+04  6.67572647e+07]]
    # Q = np.array([[ 4.03005280e-06, -3.83997884e-04, -5.90703882e+00],
    #               [-3.83997884e-04,  1.73656793e+01,  2.89105567e+04],
    #               [-5.90703882e+00,  2.89105567e+04,  6.67572647e+07]])
    # Q = np.array([[ 1e-1, -3.83997884e-04, -5.90703882e+00],
    #               [-3.83997884e-04,  1e-1,  2.89105567e+04],
    #               [-5.90703882e+00,  2.89105567e+04,  1e-1]])
    # Q = np.array([[ 4.03005280e-06, 0, 0],
    #               [0,  1.73656793e+01,  0],
    #               [0,  0,  6.67572647e+07]]) 
    #----------------------------------------------------------------#
    # USING LAE EST ACC
    # np.cov(err) =  [[1.14494042e-05 3.07764544e-03 1.62770948e-02]
    #                 [3.07764544e-03 1.53484685e+00 1.42351785e+01]
    #                 [1.62770948e-02 1.42351785e+01 1.04866619e+03]]
    # Q = np.array([[1e-1, 3.07764544e-03, 1.62770948e-02],
    #               [3.07764544e-03, 1e-1, 1.42351785e+01],
    #               [1.62770948e-02, 1.42351785e+01, 1e-1]])
    # Q = np.array([[1.14494042e-05, 3.07764544e-03, 1.62770948e-02],
    #               [3.07764544e-03, 1.53484685e+00, 1.42351785e+01],
    #               [1.62770948e-02, 1.42351785e+01, 1.04866619e+03]])
    #----------------------------------------------------------------#
    # USING CMD ACC
    # np.cov(err) =  [[1.19728937e-07 2.04532016e-06 2.04531994e-03]
    #                 [2.04532016e-06 2.71624020e-03 2.71624015e+00]
    #                 [2.04531994e-03 2.71624015e+00 2.71624010e+03]]
    # Q = np.array([[1.19728937e-07, 2.04532016e-06, 2.04531994e-03],
    #               [2.04532016e-06, 2.71624020e-03, 2.71624015e+00],
    #               [2.04531994e-03, 2.71624015e+00, 2.71624010e+03]])
    # Q = np.array([[1e-1, 2.04532016e-06, 2.04531994e-03],
    #               [2.04532016e-06, 1e-1, 2.71624015e+00],
    #               [2.04531994e-03, 2.71624015e+00, 1e-1]])
    #----------------------------------------------------------------#
    # ORIGINAL
    # Q = np.array([[1e-6, 0, 0],
    #               [0, 0.05501, 0],
    #               [0, 0, 5501*289*10**-3]]) 
    # MIN_Q_VALUE = np.array([[1e-6, 0, 0],
    #                         [0, 0.05501*10**-3, 0],
    #                         [0, 0, 5501*289*10**-6]])
    # MAX_Q_VALUE = np.array([[1e-6, 0, 0],
    #                         [0, 0.05501*10**3, 0],
    #                         [0, 0, 5501*289*10**0]])
    #----------------------------------------------------------------#
    ## IPS300_G30
    # USING LAE EST ACC 
    # np.cov(err) =  [[ 1.07560069e-04 -6.71932504e-04 -1.62890570e+02]
    #                 [-6.71932504e-04  3.74740359e+02  5.29396854e+05]
    #                 [-1.62890570e+02  5.29396854e+05  1.26481282e+09]]
    # Q = np.array([[ 1.07560069e-04, -6.71932504e-04, -1.62890570e+02],
    #               [-6.71932504e-04,  3.74740359e+02,  5.29396854e+05],
    #               [-1.62890570e+02,  5.29396854e+05,  1.26481282e+09]])
    #----------------------------------------------------------------#
    # MOUSE CFD ACC
    # np.cov(err) =  [[4.08063224e-27 3.52234601e-14 7.04550127e-11]
    #                 [3.52234601e-14 1.10921637e+04 2.21401106e+07]
    #                 [7.04550127e-11 2.21401106e+07 4.42802211e+10]]
    # Q = np.array([[4.08063224e-27, 3.52234601e-14, 7.04550127e-11],
    #                 [3.52234601e-14, 1.10921637e+04, 2.21401106e+07],
    #                 [7.04550127e-11, 2.21401106e+07, 4.42802211e+10]])
    #----------------------------------------------------------------#
    # MOUSE LAE ACC
    # np.cov(err) =  [[ 3.57026540e-03  1.60612883e+00 -1.47234703e+01]
    #                 [ 1.60612883e+00  3.23561743e+03 -1.26937496e+04]
    #                 [-1.47234703e+01 -1.26937496e+04  1.59845381e+05]]
    # Q = np.array([[ 3.57026540e-03,  1.60612883e+00, -1.47234703e+01],
    #                 [ 1.60612883e+00,  3.23561743e+03, -1.26937496e+04],
    #                 [-1.47234703e+01, -1.26937496e+04,  1.59845381e+05]])

   
    # R = 0.0126*2 # 3*10e-4 # 3000 # 5000 # 5000 #150 #500 #1000 #100 #10 #1.5 #1 #與誤差有關 -> 影響平滑度
    # R = 0.00126*2 # 3*10e-4 # 3000 # 5000 # 5000 #150 #500 #1000 #100 #10 #1.5 #1 #與誤差有關 -> 影響平滑度
    # R = 0.0022151714012120923
    R = 0.1**2
    # R = 1.1966744310659976e-07
    P = np.array([[1e-4, 0, 0],
                  [0, 1e-4, 0],
                  [0, 0, 1e-4]])
    Wt = 0
    pose = np.zeros(len(Pos))
    vele = np.zeros(len(Pos))
    acce = np.zeros(len(Pos))
    xm = np.zeros((3, 1))  
    Pm = P
    u_values = []
    u_v_values = []
    u_a_values = []
    y_values = []
    delta_x_values = []
    Q_acc = []
    Q_vel = []
    Q_pos = []
    Q_0err = []
    count_1 = 0
    count_2 = 0
    count_3 = 0
    
    for i in range(len(Pos)): # m = measurement;p = predict len(pos)
        Pp = np.dot(np.dot(A, Pm), A.T) + Q
        xp = np.dot(A, xm) + Wt
        # xp = A@xm + Wt
        # Wt = xp - np.dot(A, xm)
        # xp = np.dot(A, xm) + 1e0 * np.dot(B, u[i]) + Wt 
        Km = np.dot(Pp, C.T) / (np.dot(np.dot(C, Pp), C.T) + R)
        # z = np.array([[Pos[i]],
        #               [Vel[i]],
        #               [Acc_CFD_est[i]]])
        # y = (z - np.dot(C, xp))
        y = (Pos[i] - np.dot(C, xp))
        # 求y_v
        # y_v = (-3.83997884e-04 / 4.03005280e-06**2) * y_p
        xm = xp + np.dot(Km, y) # =x_hat
        Pm = np.dot((np.eye(3) - np.dot(Km, C)), Pp)

        ##-------加速度求Q-------##
        print("-------加速度求Q-------")
        ## 求M
        u_p = (Pos[i] - xm[0]) # y(k)-x_hat(k)
        # u_a = (2.04531994e-03 / 1.19728937e-07) * u_p # cmd
        # u_a = (-5.90703882e+00 / 4.03005280e-06) * u_p # CFD
        # u_a = (1.62770948e-02 /1.14494042e-05) * u_p # LAE
        # u_a = (-1.62890570e+02 / 1.07560069e-04) * u_p # LAE IPS300G30
        u_a = (5.95731020e+04/1.99076374e-02) * u_p # sim data
        # u_a = (AccCmd[i] - xm[2]) 
        print("u_a = ", u_a)
        # u = Acc_CFD_est[i] - xm[2]
        print("xm[2] = ", xm[2])
        u_a_values.append(u_a)
        # y_values.append(y)
        n = i+1
        u_sqr = [val**2 for val in u_a_values[:n]]
        M = sum(u_sqr[:n])/n
        print("M = ", M)
        ## 求G_telda
        Y = np.dot(np.dot(C, Pp), C.T) + R
        print("Y = ", Y)
        print("Km = ", Km)
        G_tel = Km[2]**2 * Y
        # G_tel = np.dot(np.dot(Km.T, Km), Y)
        print("G_tel = ", G_tel)
        ## 求G_hat
        delta_x = xm[2] - xp[2] # x - x-
        delta_x_values.append(delta_x)
        print("delta_x = ", delta_x)
        m = i+1
        # delta_x_values_sqr = [np.dot(val.T, val) for val in delta_x_values[:m]]
        delta_x_values_sqr = [val**2 for val in delta_x_values[:m]]
        # print("sum(delta_x_values_sqr[:m]) = ", sum(delta_x_values_sqr[:m]))
        G_hat = sum(delta_x_values_sqr[:m]) / m
        print("G_hat = ", G_hat)
        ## 求S
        G = G_hat/G_tel
        S = np.maximum(1, G_hat/G_tel)
        print("G_hat/G_tel = ", G_hat/G_tel)
        print("S = ", S)
        ## 求Q_hat
        # Q_hat = np.dot(S, M)
        Q_hat = S * M
        a = Q_hat
        print("Q_hat = ", Q_hat)
        # Q_hat_array = np.array([1, 1, Q_hat[0, 0]])  # Extracting the scalar value from 2-D array
        # Q_hat_matrix = np.diag(Q_hat_array)
        Q_hat_matrix = Q
        Q_hat_matrix[2, 2] = Q_hat[0, 0]
        print("Q_hat_matrix = ", Q_hat_matrix)
        ## 更新Q值
        # print("Q = ", Q)
        # Q_new = np.dot(Q, Q_hat_matrix)
        # if (Q_new[2, 2] > MAX_Q_VALUE[2, 2] or Q_new[2, 2] < MIN_Q_VALUE[2, 2]):
        #     Q_new[2, 2] = np.random.uniform(MAX_Q_VALUE[2, 2], MIN_Q_VALUE[2, 2])
        #     print("Q_new[2, 2] = ", Q_new[2, 2])
        #     count_1 = count_1 + 1
        # print("count_1 = ", count_1)
        Q_new = Q_hat_matrix
        # print("Q_new = ", Q_new)
        # udate_factor = 0.5
        # Q_new = Q + (Q_new - Q) * udate_factor
        Q = Q_new
        Q_acc.append(Q_new[2, 2])
        # MIN_Q_VALUE[2, 2] = np.mean(Q_acc)
        print("--------------------------------------------------")

        ##-------速度求Q-------##
        print("-------速度求Q-------")
        ## 求M
        u_p = (Pos[i]- xm[0])
        # u_v = (2.04532016e-06 / 1.19728937e-07) * u_p # CMD
        # u_v = (-3.83997884e-04 / 4.03005280e-06) * u_p # CFD
        # u_v = (3.07764544e-03 /1.14494042e-05) * u_p # LAE
        # u_v = (-6.71932504e-04 /1.07560069e-04) * u_p # LAE IPS300G30
        u_v = (2.97873450e+01/1.99076374e-02) * u_p # sim data
        # u_v = VelCmd[i]- xm[1]
        print("u_v = ", u_v)
        print("xm[1] = ", xm[1])
        u_v_values.append(u_v)
        # y_values.append(y)
        n = i+1
        u_sqr = [val**2 for val in u_v_values[:n]]
        M = sum(u_sqr[:n])/n
        print("M = ", M)
        ## 求G_tel
        G_tel = Km[1]**2 * Y
        # G_tel = np.dot(np.dot(Km.T, Km), Y)
        print("G_tel = ", G_tel)
        ## 求G_hat
        delta_x = xm[1] - xp[1] # x - x-
        delta_x_values.append(delta_x)
        print("delta_x = ", delta_x)
        m = i+1
        # delta_x_values_sqr = [np.dot(val.T, val) for val in delta_x_values[:m]]
        delta_x_values_sqr = [val**2 for val in delta_x_values[:m]]
        # print("sum(delta_x_values_sqr[:m]) = ", sum(delta_x_values_sqr[:m]))
        G_hat = sum(delta_x_values_sqr[:m]) / m
        print("G_hat = ", G_hat)
        ## 求S
        G = G_hat/G_tel
        S = np.maximum(1, G_hat/G_tel)
        print("G_hat/G_tel = ", G_hat/G_tel)
        print("S = ", S)
        ## 求Q_hat
        Q_hat = S * M
        a = Q_hat
        print("Q_hat = ", Q_hat)
        # Q_hat_array = np.array([1, Q_hat[0, 0], 1])  # Extracting the scalar value from 2-D array
        # Q_hat_matrix = np.diag(Q_hat_array)
        Q_hat_matrix = Q
        Q_hat_matrix[1, 1] = Q_hat[0, 0]
        print("Q_hat_matrix = ", Q_hat_matrix)
        ## 更新Q值
        # print("Q = ", Q)
        # Q_new = np.dot(Q, Q_hat_matrix)
        # if (Q_new[1, 1] > MAX_Q_VALUE[1, 1] or Q_new[1, 1] < MIN_Q_VALUE[1, 1]):
        #     Q_new[1, 1] = np.random.uniform(MAX_Q_VALUE[1, 1], MIN_Q_VALUE[1, 1])
        #     print("Q_new[1, 1] = ", Q_new[1, 1])
        #     count_2 = count_2 + 1
        # print("count_2 = ", count_2)
        Q_new = Q_hat_matrix
        # print("Q_new = ", Q_new)
        # update_factor = 0.5
        # Q_new = Q + (Q_new - Q) * update_factor
        Q = Q_new
        Q_vel.append(Q_new[1, 1])
        # MIN_Q_VALUE[1, 1] = np.mean(Q_vel)
        print("--------------------------------------------------")

        ##-------位置求Q-------##
        print("-------位置求Q-------")
        ## 求M
        u = Pos[i]- xm[0]
        # u_save.append(u)
        print("u = ", u)
        # u = Vel[i]- xm[1]
        print("xm[0] = ", xm[0])
        u_values.append(u)
        # y_values.append(y)
        n = i+1
        u_sqr = [val**2 for val in u_values[:n]]
        M = sum(u_sqr[:n])/n
        print("M = ", M)
        ## 求G_tel
        G_tel = Km[0]**2 * Y
        # G_tel = np.dot(np.dot(Km.T, Km), Y)
        print("G_tel = ", G_tel)
        ## 求G_hat
        delta_x = xm[0] - xp[0] # x - x-
        delta_x_values.append(delta_x)
        print("delta_x = ", delta_x)
        m = i+1
        # delta_x_values_sqr = [np.dot(val.T, val) for val in delta_x_values[:m]]
        delta_x_values_sqr = [val**2 for val in delta_x_values[:m]]
        # print("sum(delta_x_values_sqr[:m]) = ", sum(delta_x_values_sqr[:m]))
        G_hat = sum(delta_x_values_sqr[:m]) / m
        print("G_hat = ", G_hat)
        ## 求S
        G = G_hat/G_tel
        S = np.maximum(1, G_hat/G_tel)
        print("G_hat/G_tel = ", G_hat/G_tel)
        print("S = ", S)
        ## 求Q_hat
        Q_hat = S * M
        a = Q_hat
        print("Q_hat = ", Q_hat)
        # Q_hat_array = np.array([Q_hat[0, 0], 1, 1])  # Extracting the scalar value from 2-D array
        # Q_hat_matrix = np.diag(Q_hat_array)
        Q_hat_matrix = Q
        Q_hat_matrix[0, 0] = Q_hat[0, 0]
        print("Q_hat_matrix = ", Q_hat_matrix)
        ## 更新Q值
        # print("Q = ", Q)
        # Q_new = np.dot(Q, Q_hat_matrix)
        # if (Q_new[0, 0] > MAX_Q_VALUE[0, 0] or Q_new[0, 0] < MIN_Q_VALUE[0, 0]):
        #     Q_new[0, 0] = np.random.uniform(MAX_Q_VALUE[0, 0], MIN_Q_VALUE[0, 0])
        #     print("Q_new[0, 0] = ", Q_new[0, 0])
        #     count_3 = count_3 + 1
        # print("count_3 = ", count_3)
        Q_new = Q_hat_matrix
        # print("Q_new = ", Q_new)
        # update_factor = 0.01
        # Q_new = Q + (Q_new - Q) * update_factor
        Q = Q_new
        Q_pos.append(Q_new[0, 0])
        if u <= 0.000005:
            Q_0err.append(Q_hat_matrix)

        # MIN_Q_VALUE[1, 1] = np.mean(Q_vel)
        print("--------------------------------------------------")

        pose[i] = xm[0]
        vele[i] = xm[1]
        acce[i] = xm[2]
    # # Plot Q
    # plt.figure()
    # # plt.plot(range(1, len(Q_acc) + 1), Q_acc, color='r', label="ACC_Q")
    # plt.scatter(range(1, len(Q_vel) + 1), Q_vel, color='b', label="VEL_Q")
    # plt.xlabel('Iteration')
    # plt.ylabel('Q Values')
    # plt.title('Q Values over Iterations')
    # plt.legend(loc="upper right")
    # plt.grid()

    # plt.figure()
    # plt.scatter(range(1, len(Q_acc) + 1), Q_acc, color='r', label="ACC_Q")
    # # plt.plot(range(1, len(Q_vel) + 1), Q_vel, color='b', label="VEL_Q")
    # plt.xlabel('Iteration')
    # plt.ylabel('Q Values')
    # plt.title('Q Values over Iterations')
    # plt.legend(loc="upper right")
    # plt.grid()
    # plt.show()
    return pose, vele, acce, Q_pos, Q_acc, Q_vel, u_values, Q_0err