import numpy as np
import mousedata_add
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

def Cal(Mousedata, Motordata, SamplingTime, CPI) :
    # Extracting data from Motordata
    Pos = np.array(Motordata[:, 3],float)
    PosCmd = np.array(Motordata[:, 4],float)
    Vel = np.array(Motordata[:, 5],float)
    VelCmd = np.array(Motordata[:, 6],float)
    TorCtrl = np.array(Motordata[:, 7],float)
    AccCmd = np.array(Motordata[:, 8],float)
    t = np.arange(0, len(Motordata[:,0]) * SamplingTime, SamplingTime)
    # 時間轉秒
    Motordata = np.hstack((Motordata, np.zeros((Motordata.shape[0], 1)))) # 增加一列來計算時間
    for i in range(1, len(Motordata)):
        fmt = '%H:%M:%S.%f'
        time1 = datetime.strptime(Motordata[i-1, 1], fmt).time()
        total_seconds = time1.hour * 3600 + time1.minute * 60 + time1.second + time1.microsecond / 1e6 #  time1.minute * 60 +
        Motordata[i-1 , 9] = total_seconds
    # Processing mouse data
    mousedata_data = mousedata_add.mousedata_add(Mousedata, Mousedata)
    mouseX = np.array(abs(mousedata_data[:, 0]),float)
    mouseY = np.array(mousedata_data[:, 1],float)
    mouse_displacement = mousedata_data[:, 5].astype(float)
    # print(len(mouse_displacement))
    MotorTime = np.array(Motordata[:, 9],float) - min(np.array(mousedata_data[0, 6], float), np.array(Motordata[0, 9],float))
    MouseTime = np.array(mousedata_data[:, 6],float) - min(np.array(mousedata_data[0, 6], float), np.array(Motordata[0, 9],float))
    # Interpolating mouse displacement
    mouse_real_Pos = interp1d(np.array(mousedata_data[:, 2],float), mouse_displacement, fill_value="extrapolate")(t) / CPI #得到inch
    return  MouseTime, MotorTime, mouseX, mouseY, Pos, PosCmd, Vel, VelCmd, AccCmd, TorCtrl, mousedata_data, mouse_displacement, mouse_real_Pos