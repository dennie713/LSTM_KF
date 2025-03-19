import numpy as np
from scipy.signal import butter, filtfilt

def zero_phase_filter(Order, CutoffFreq, data): 
    # Calculate sampling frequency
    SamplingTime = 0.001
    fm = 1 / SamplingTime
    # Butterworth 低通滤波器，正则化：CutoffFreq / ( Sampling / 2 )，返回传递函数的系数
    b, a = butter(Order, CutoffFreq / (fm / 2), 'low')
    
    # 零相位数字滤波器：正向和反向各滤波一次，以消除零相位失真
    filtered_data = filtfilt(b, a, data) #濾波後的數值
    return filtered_data