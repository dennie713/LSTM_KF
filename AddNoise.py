import numpy as np
import random
from scipy.interpolate import interp1d

# 加入雜訊
def AddNoice(motor_data, motor_data_Cmd, mouse_real_data) :

    noice = 0.0001 #每筆數據的+-1%
    noice_percent = np.zeros(len(motor_data))
    motor_data_added_noice = []
    motor_data_Cmd_added_noice = []
    mouse_real_data_added_noice = []
    noice_percent_record = []

    for i in range(len(motor_data)):
        # noice_percent = random.choice(noice_range)*percent
        noice_percent[i] = random.randint(-10,10) * noice
        motor_data_added_noice.append(motor_data[i] + (noice_percent[i]))
        motor_data_Cmd_added_noice.append(motor_data_Cmd[i] + (noice_percent[i]))# + noice_percent
        mouse_real_data_added_noice.append(mouse_real_data[i] + (noice_percent[i]))
        noice_percent_record.append(noice_percent[i])
    
    motor_data_added_noice = np.array(motor_data_added_noice)    
    motor_data_Cmd_added_noice = np.array(motor_data_Cmd_added_noice)
    mouse_real_data_added_noice = np.array(mouse_real_data_added_noice)
    dev  = np.mean(motor_data_Cmd_added_noice - motor_data)
    # print('ave_dev of motor_data_Cmd_added_noice :', dev)
    return motor_data_added_noice, motor_data_Cmd_added_noice, mouse_real_data_added_noice, noice_percent_record