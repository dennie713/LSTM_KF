import matplotlib.pyplot as plt
import numpy as np
import sys
import mousedata_add,  ImportData
import Cal, CFD, KF_v2, zero_phase_filter

# MAIN
if __name__ == "__main__":
    # Constant
    SamplingTime = 0.001
    CPI = 1600
    ## 木盤半徑12.5 壓克力盤半徑12.53
    wood = 12.5
    plastic = 11.945 #11.62 #11.14 # 12.53
    
    ## 量測半徑
    r = 11.6287
    ## 讀取檔案
    path1 = ['Raw_Data/IPS650_G50_F_motion.txt'] #馬達資料.txt路徑
    path2 = ['Raw_Data/IPS650_G50_F_mouse.txt']  #滑鼠資料.txt路徑
    # path1 = ['build/IPS200_G26_FBFB_motion.txt'] #馬達資料.txt路徑
    # path2 = ['build/IPS200_G26_FBFB_mouse.txt']  #滑鼠資料.txt路徑
    
    Motordata, Mousedata = ImportData.ImportData(path1, path2)
    MouseTime, MotorTime, mouseX, mouseY, Pos, PosCmd, Vel, VelCmd, AccCmd, TorCtrl, mousedata_data, mouse_displacement, mouse_real_Pos = Cal.Cal(Mousedata, Motordata, SamplingTime, CPI) 
    t = np.arange(0, (len(Motordata[:,0])) * SamplingTime, SamplingTime)

    # print(mouse_displacement)

    # 將資料濾波
    filtered_Pos = zero_phase_filter.zero_phase_filter(3, 17, Pos)
    filtered_PosCmd = zero_phase_filter.zero_phase_filter(3, 50, PosCmd)
    filtered_mouse_real_Pos = zero_phase_filter.zero_phase_filter(3, 17, mouse_real_Pos)
    Pos = filtered_Pos
    PosCmd = filtered_PosCmd
    mouse_real_Pos = filtered_mouse_real_Pos

    ## CFD 速度&加速度
    Pos_CFD_est, Vel_CFD_est, Acc_CFD_est = CFD.CFD(Pos) 
    Pos_CFD_Cmd, Vel_CFD_Cmd, Acc_CFD_Cmd = CFD.CFD(PosCmd) 
    Pos_CFD_est_mouse, Vel_CFD_est_mouse, Acc_CFD_est_mouse = CFD.CFD(mouse_real_Pos) 
    # Acc_CFD_est_mouse = CFD_2.CFD_2(Vel_CFD_est_mouse, Pos)

    ## KF 速度&加速度
    Pos_KF_est, Vel_KF_est, Acc_KF_est = KF_v2.KF_ORIG(0.001, Pos, Acc_CFD_est)
    SIGMA = 0
    Pos_KF_est_mouse, Vel_KF_est_mouse, Acc_KF_est_mouse = KF_v2.KF(0.001, mouse_real_Pos, Acc_CFD_est, SIGMA)
    
    # SIGMA_LAE = 362597
    # Pos_KF_est1, Vel_KF_est1, Acc_KF_est1 = KF_v2.KF(0.001, Pos, Acc_CFD_est, SIGMA_LAE)
    # Pos_KF_est_mouse1, Vel_KF_est_mouse1, Acc_KF_est_mouse1 = KF_v2.KF(0.001, mouse_real_Pos, Acc_CFD_est, SIGMA_LAE)
    
    # SIGMA_CMD = 472964
    # Pos_KF_est2, Vel_KF_est2, Acc_KF_est2 = KF_v2.KF(0.001, Pos, Acc_CFD_est, SIGMA_CMD)
    # Pos_KF_est_mouse2, Vel_KF_est_mouse2, Acc_KF_est_mouse2 = KF_v2.KF(0.001, mouse_real_Pos, Acc_CFD_est, SIGMA_CMD)
    
    # SIGMA_CFD = 20510470.13349476
    # Pos_KF_est3, Vel_KF_est3, Acc_KF_est3 = KF_v2.KF(0.001, Pos, Acc_CFD_est, SIGMA_CFD)
    # Pos_KF_est_mouse3, Vel_KF_est_mouse3, Acc_KF_est_mouse3 = KF_v2.KF(0.001, mouse_real_Pos, Acc_CFD_est, SIGMA_CFD)

    ##選擇估測方法
    Pos_est = Pos_KF_est
    Vel_est = Vel_KF_est
    Acc_est = Acc_KF_est
    Pos_est_mouse = Pos_KF_est_mouse
    Vel_est_mouse = Vel_KF_est_mouse
    Acc_est_mouse = Acc_KF_est_mouse

    ##為了正反轉方向計算
    #檢查path中是否包含 "FB","BF","FBFB"，找出有正反轉(方向有變化的)
    keywords = ["B", "FB", "BF", "FBFB"]
    contains_string = any(keyword in path for path in path1 for keyword in keywords)
    if contains_string:
        print("有反轉方向")
        for i in range(1, len(Vel_est)):
            if Vel_est[i] < 0: 
                Vel_est_mouse[i] = -Vel_est_mouse[i]
        Vel_est_mouse = zero_phase_filter.zero_phase_filter(3, 19, Vel_est_mouse)
    else:
        print("無反轉方向")

    # # 位置比较图
    # plt.figure(11)
    # plt.plot(t, Pos_est, label="Motor Position", linewidth=2)
    # plt.title("Motor and Mouse Position Comparison")
    # plt.xlabel("Time (sec)")
    # plt.ylabel("Motor Position (rad)")
    # plt.plot(t, Pos_est_mouse, 'r', label="Mouse Position", linewidth=1)
    # plt.legend(loc="lower center")

    # 速度比较图
    plt.figure(12)
    Vel_est = [x*r/2.54 for x in Vel_est] 
    # Vel_KF_est1 = [x*r/2.54 for x in Vel_KF_est1]
    # Vel_KF_est2 = [x*r/2.54 for x in Vel_KF_est2]
    # Vel_KF_est3 = [x*r/2.54 for x in Vel_KF_est3]
    # Vel_est_mouse_real = Vel_est_mouse
    
    # plt.plot(t, Vel_KF_est1, 'orange', label="Motor Velocity LAE", linewidth=2, linestyle='-')
    # plt.plot(t, Vel_KF_est1, 'red', label="Motor Velocity CMD", linewidth=1, linestyle='-')
    # plt.plot(t, Vel_KF_est3, 'green', label="Motor Velocity CFD", linewidth=1, linestyle='-')
    plt.plot(t, Vel_est, 'blue', label="Motor Velocity", linewidth=1, linestyle='-')
    plt.title("Motor and Mouse Velocity Comparison")
    # plt.title("Mouse Velocity")
    plt.xlabel("Time (sec)")
    plt.yticks(np.arange(0, 1000, 50))
    
    # plt.plot(t, Vel_KF_est_mouse1, 'orange', label="Mouse Velocity LAE", linewidth=2)
    # plt.plot(t, Vel_KF_est_mouse2, 'red', label="Mouse Velocity CMD", linewidth=1)
    # plt.plot(t, Vel_KF_est_mouse3, 'green', label="Mouse Velocity CFD", linewidth=1)
    plt.plot(t, Vel_est_mouse, 'red', label="Mouse Velocity", linewidth=1)
    plt.ylabel("Velocity (IPS)")
    # plt.legend(loc="lower center")
    plt.legend()
    plt.grid()

    # 加速度比较图
    plt.figure(13)
    Acc_est = [x*r*0.01/9.81 for x in Acc_est]
    # Acc_KF_est1 = [x*r*0.01/9.81 for x in Acc_KF_est1]
    # Acc_KF_est2 = [x*r*0.01/9.81 for x in Acc_KF_est2]
    # Acc_KF_est3 = [x*r*0.01/9.81 for x in Acc_KF_est3]
    if contains_string:
        Acc_est_mouse = CFD.CFD_2(Vel_est_mouse)
    Acc_est_mouse[(Acc_est_mouse < -70 / 0.0254 * 9.81) | (Acc_est_mouse >  70 / 0.0254 * 9.81)] = 0
    Acc_est_mouse = [x*0.0254/ 9.81 for x in Acc_est_mouse]
    # Acc_KF_est_mouse1 = [x*0.0254/ 9.81 for x in Acc_KF_est_mouse1]
    # Acc_KF_est_mouse2 = [x*0.0254/ 9.81 for x in Acc_KF_est_mouse2]
    # Acc_KF_est_mouse3 = [x*0.0254/ 9.81 for x in Acc_KF_est_mouse3]
    # Acc_est_mouse_real = Acc_est_mouse 
    # plt.plot(t, Acc_KF_est1, 'orange', label="Motor Acceleration LAE", linewidth=2)
    # plt.plot(t, Acc_KF_est1, 'red', label="Motor Acceleration CMD", linewidth=1)
    # plt.plot(t, Acc_KF_est3, 'green', label="Motor Acceleration CFD", linewidth=1)
    plt.plot(t, Acc_est, 'blue', label="Motor Acceleration", linewidth=1)
    plt.title("Motor and Mouse Acceleration Comparison")
    # plt.title("Mouse Acceleration")
    plt.xlabel("Time (sec)")
    plt.yticks(np.arange(-60, 60, 10))
    # plt.plot(t, Acc_KF_est_mouse1, 'orange', label="Mouse Acceleration LAE", linewidth=2)
    # plt.plot(t, Acc_KF_est_mouse2, 'red', label="Mouse Acceleration CMD", linewidth=1)
    # plt.plot(t, Acc_KF_est_mouse3, 'green', label="Mouse Acceleration CFD", linewidth=1)
    plt.plot(t, Acc_est_mouse, 'red', label="Mouse Acceleration", linewidth=1)
    plt.ylabel("MAcceleration (G)")
    # plt.legend(loc="lower center")
    plt.legend()
    plt.grid()

    # # 速度误差
    # plt.figure(14)
    # deviation = np.abs(Vel_est_mouse - Vel_est) / np.where(Vel_est != 0, Vel_est, 1) * 100
    # deviation[(deviation < -100) | (deviation > 100)] = 0
    # # print("mean error = ", np.mean(np.abs(deviation)))
    # # print(min(np.abs(deviation)))
    # plt.plot(t, np.abs(deviation))
    # abs_deviation = np.abs(deviation)
    # plt.title("Velocity Error")
    # plt.xlabel("Time (sec)")
    # plt.ylabel("Error (%)")

    plt.show()
