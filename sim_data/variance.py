import numpy as np
# import mousedata_add
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import sys
# import mousedata_add,  ImportData
import LAE, CFD, LSF, KF_v2, zero_phase_filter

#--------------------------------------------------sim data processing------------------------------------------------#
path1 = ['sim_data/raw_data_all.txt'] #馬達資料.txt路徑
SamplingTime = 0.001
for i in range (len(path1)):
    #Simdata
    Simdata = []
    with open(path1[i], 'r')as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            Simdata.append(line) #模擬資料

Simdata = np.array(Simdata, dtype=np.float64)

pos_data = Simdata[:, 0]
pos_cmd_data = Simdata[:, 1]
vel_cmd_data = Simdata[:, 2]
acc_cmd_data = Simdata[:, 3]

Pos_CFD_est, Vel_CFD_est, Acc_CFD_est = CFD.CFD(pos_data) 
Vel_LSF_est = LSF.LSF14(pos_data)
Acc_LAE_est = LAE.LAE(pos_data, 0.001)

zk = []
err = []

for i in range(len(pos_data)-1):
    dt = 0.001
    zk = np.array([[pos_cmd_data[i]], 
                    [vel_cmd_data[i]],
                    [Acc_CFD_est[i]]]) # vel為LSF1/4
    zk_1 = np.array([[pos_cmd_data[i+1]], 
                    [vel_cmd_data[i+1]],
                    [Acc_CFD_est[i+1]]]) # vel為LSF1/4
    A = np.array([[1, dt, 0.5*dt**2],
                  [0, 1, dt],
                  [0, 0, 1 ]])
    # print("zk = ", zk)
    # print("zk_1 = ", zk_1)
    # print("A =", A)
    # print("A*zk = ", A @ zk)
    error = zk_1 - A @ zk
    err.append(error)
err = np.array(err).squeeze()

print('-----------------------------------------------------------------')
# print(err)
print("np.cov(err) = \n", np.cov(err.T)) 
# print("zk = ", zk)
# print("zk_1 = ", zk_1)
# print("A*zk = ", A @ zk)
# print("error = ", error)
variance_p = np.var(err[:, 0])
print("variance_p = ", variance_p)
variance_v = np.var(err[:, 1])
print("variance_v = ", variance_v)
variance_a = np.var(err[:, 2])
print("variance_a = ", variance_a)
print('-----------------------------------------------------------------')
print("acc estimate")
print("acc CMD variance :", np.var(acc_cmd_data))
print("acc CFD variance :", np.var(Acc_CFD_est))
print("acc LAE variance :", np.var(Acc_LAE_est))
print('-----------------------------------------------------------------')