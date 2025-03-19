import numpy as np
from scipy.signal import butter, filtfilt

def zero_phase_filter(Order, CutoffFreq, data): 
    # Calculate sampling frequency
    SamplingTime = 0.001
    fm = 1 / SamplingTime
    # Butterworth 低通濾波器，正則化：CutoffFreq / ( Sampling / 2 )，返回傳遞函數的係數
    b, a = butter(Order, CutoffFreq / (fm / 2), 'low')
    
    # 零相位數位濾波器：正向和反向各濾波一次，以消除零相位失真
    filtered_data = filtfilt(b, a, data) #濾波後的數值
    return filtered_data