import matplotlib.pyplot as plt
import numpy as np
import sys
import KF_v2, zero_phase_filter

# MAIN
if __name__ == "__main__":
    
    path1 = ['sim_data/raw_data_all.txt'] #馬達資料.txt路徑
    SamplingTime = 0.001
    for i in range (len(path1)): # len(path1)
        #Simdata
        Simdata = []
        with open(path1[i], 'r')as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                Simdata.append(line) #模擬資料

    Simdata = np.array(Simdata, dtype=np.float64)
    # pos_data = Simdata[:, 0]
    # pos_cmd_data = Simdata[:, 1]
    # vel_cmd_data = Simdata[:, 2]
    # acc_cmd_data = Simdata[:, 3]
    # 取前200筆
    pos_data = Simdata[:201, 0]
    pos_cmd_data = Simdata[:201, 1]
    vel_cmd_data = Simdata[:201, 2]
    acc_cmd_data = Simdata[:201, 3]

    # acc CMD variance : 1.1049662693063255e-05
    # acc CFD variance : 15656046016.025362
    # acc LAE variance : 15517401.657680077
    Pos_KF_CMD, Vel_KF_CMD, Acc_KF_CMD = KF_v2.KF(0.001, pos_data, 1.1049662693063255e-05)
    Pos_KF_CFD, Vel_KF_CFD, Acc_KF_CFD = KF_v2.KF(0.001, pos_data, 15656046016.025362)
    Pos_KF_LAE, Vel_KF_LAE, Acc_KF_LAE = KF_v2.KF(0.001, pos_data, 15517401.657680077)
    Pos_KF, Vel_KF, Acc_KF = KF_v2.KF_ORIG(0.001, pos_data)
    
    # pos
    plt.figure()
    plt.plot(pos_cmd_data, "black", label="true CMD", linewidth=1)
    plt.plot(pos_data, "red", label="true CMD addNoise", linewidth=1)
    plt.plot(Pos_KF, "blue", label="try&err Q", linewidth=1)
    plt.plot(Pos_KF_CMD, "orange", label="KF CMD Q", linewidth=1)
    plt.plot(Pos_KF_CFD, "cyan", label="KF CFD Q", linewidth=1)
    plt.plot(Pos_KF_LAE, "purple", label="KF LAE Q", linewidth=1)
    plt.title("Pos Comparison")
    plt.xlabel("Time")
    plt.ylabel("Pos")
    plt.legend()

    # vel
    plt.figure()
    plt.plot(vel_cmd_data, "black", label="true CMD", linewidth=1)
    # plt.plot(Vel_KF, "blue", label="try&err Q", linewidth=1)
    plt.plot(Vel_KF_CMD, "orange", label="KF CMD Q", linewidth=1)
    plt.plot(Vel_KF_CFD, "cyan", label="KF CFD Q", linewidth=1)
    plt.plot(Vel_KF_LAE, "purple", label="KF LAE Q", linewidth=1)
    plt.title("Vel Comparison")
    plt.xlabel("Time")
    plt.ylabel("Vel")
    plt.legend()

    # acc
    plt.figure()
    plt.plot(acc_cmd_data, "black", label="true CMD", linewidth=1)
    # plt.plot(Acc_KF, "blue", label="try&err Q", linewidth=1)
    plt.plot(Acc_KF_CMD, "orange", label="KF CMD Q", linewidth=1)
    plt.plot(Acc_KF_CFD, "cyan", label="KF CFD Q", linewidth=1)
    plt.plot(Acc_KF_LAE, "purple", label="KF LAE Q", linewidth=1)
    plt.title("Acc Comparison")
    plt.xlabel("Time")
    plt.ylabel("Acc")
    plt.legend()

    plt.show()


