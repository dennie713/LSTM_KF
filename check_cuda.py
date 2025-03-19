import numpy as np
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import motor.ImportData as ImportData, motor.Cal as Cal
import KF, KF_sim, LSTM
# 強制使用CPU
# device = torch.device("cpu") 
# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

print(f"Using device: {device}")
print(torch.__version__)
print(torch.version.cuda)