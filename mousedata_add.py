# mousedata_add
import numpy as np
from datetime import datetime, timedelta

def mousedata_add(mousedata_data, Mousedata):
    fmt = '%H:%M:%S.%f'
    # Initialize output array
    mousedata_data = np.zeros((len(Mousedata),7))
    for ii in range(1, len(Mousedata)):
        time1 = datetime.strptime(Mousedata[ii-1, 1], fmt).time()
        time2 = datetime.strptime(Mousedata[ii, 1], fmt).time()
        # diff = time2 - time1
        diff_seconds = (datetime.combine(datetime.min, time2) - datetime.combine(datetime.min, time1)).total_seconds()
        # seconds_diff = diff.total_seconds()
        # new_time = time1 + diff_seconds  # 新的時間
        new_datetime = datetime.combine(datetime.min, time1) + timedelta(seconds=diff_seconds)
        new_time = new_datetime.strftime(fmt)
        # 把時間轉換成秒
        MouseTime =  time1.hour * 3600 + time1.minute * 60 + time1.second + time1.microsecond / 1e6 # time1.minute * 60

        mousedata_data [ii, 0] = Mousedata[ii, 3]
        mousedata_data [ii, 1] = Mousedata[ii, 4]
        mousedata_data [ii, 2] =  mousedata_data [ii-1, 2] + diff_seconds
        mousedata_data [ii, 3] = float(Mousedata[ii, 3]) - float(Mousedata[ii-1, 3])  # dx
        mousedata_data [ii, 4] = float(Mousedata[ii, 4]) - float(Mousedata[ii-1, 4])  # dy
        mousedata_data [ii, 5] = mousedata_data[ii-1, 5] + ((mousedata_data [ii, 3]**2 + mousedata_data [ii, 4]**2)**0.5)  # dL=(dx*2+dy*2)**2    
        mousedata_data [ii-1, 6] = MouseTime
    return mousedata_data 