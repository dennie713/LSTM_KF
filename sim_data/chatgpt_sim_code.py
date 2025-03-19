import numpy as np
import matplotlib.pyplot as plt

# 弦波參數
A = 1.0          # 振幅
omega = 2 * np.pi # 角頻率（1 Hz）
dt = 0.001        # 時間步長
t = np.arange(0, 1, dt)
true_pos = A * np.sin(omega * t)      # 真實位置
true_vel = A * omega * np.cos(omega * t) # 真實速度
true_acc = -A * omega**2 * np.sin(omega * t) # 真實加速度

# 添加測量噪聲
measurement_noise = 0.08
true_pos_addNoise = true_pos + np.random.normal(0, measurement_noise, len(true_pos))

# 狀態轉移矩陣
A_mat = np.array([
    [1, dt, 0.5 * dt**2],
    [0, 1, dt],
    [0, 0, 1]
])

# 觀測矩陣
C = np.array([[1, 0, 0]])  # 測量僅觀測位置

# 系統噪聲協方差矩陣 Q
sigma_a2 = 0.1 * (A * omega**2)**2
Q = sigma_a2 * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
                            [dt**3 / 2, dt**2, dt],
                            [dt**2 / 2, dt, 1]])
# Q = sigma_a2 * np.array([[dt**5/20, dt**4/8, dt**3/6],
#                             [dt**4/8, dt**3/3, dt**2/2],
#                             [dt**3/6, dt**2/2, dt]])
# Q = sigma_a2 * np.array( [[ 0.01,  0.01, 0.01],
#                             [ 0.01,  0.01,  0.01],
#                             [0.01,  0.01,  0.01]])
# Q = np.array( [[ 7.09368306e-04,  4.73461788e-01, -3.80660676e-01],
#                     [ 4.73461788e-01,  8.71775352e+02,  1.02284107e+03],
#                     [-3.80660676e-01,  1.02284107e+03,  1.44587281e+05]])

# 不能用的Q
# Q = 0.001223683523833341 * np.array([[dt**5/20, dt**4/8, dt**3/6],
#                            [dt**4/8, dt**3/3, dt**2/2],
#                            [dt**3/6, dt**2/2, dt]]) # CFD
# Q = 35.05406011659647 * np.array([[dt**5/20, dt**4/8, dt**3/6],
#                        [dt**4/8, dt**3/3, dt**2/2],
#                        [dt**3/6, dt**2/2, dt]]) # LAE
# Q = 0.001223683523833341 * np.array([[dt**5/20, dt**4/8, dt**3/6],
#                        [dt**4/8, dt**3/3, dt**2/2],
#                        [dt**3/6, dt**2/2, dt]]) # CMD


# 測量噪聲方差
R = np.array([[measurement_noise**2]])

# 初始條件
P = np.eye(3) * 1e-3  # 初始協方差
x = np.zeros((3, 1))  # 初始狀態 [位置, 速度, 加速度]

# 卡爾曼濾波器
est_pos = []
est_vel = []
est_acc = []

for z in true_pos_addNoise:
    # 預測
    x = np.dot(A_mat, x)
    P = np.dot(A_mat, np.dot(P, A_mat.T)) + Q

    # 更新
    K = np.dot(P, C.T) / (np.dot(C, np.dot(P, C.T)) + R)
    y = z - np.dot(C, x)  # 測量殘差
    x = x + np.dot(K, y)
    P = np.dot(np.eye(3) - np.dot(K, C), P)

    est_pos.append(x[0, 0])
    est_vel.append(x[1, 0])
    est_acc.append(x[2, 0])

# x_data_all = np.concatenate()
# raw_data_all = np.concatenate()

# 可視化結果
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, true_pos, "black", label="True Position")
plt.plot(t, true_pos_addNoise, "blue", label="true_pos_addNoise", linestyle="dotted")
plt.plot(t, est_pos, "red", label="KF Estimated Position", linewidth=1)
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(t, true_vel, "black", label="True Velocity")
plt.plot(t, est_vel, "red", label="KF Estimated Velocity", linewidth=1)
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(t, true_acc, "black", label="True Acceleration")
plt.plot(t, est_acc, "red", label="KF Estimated Acceleration", linewidth=1)
plt.legend()
plt.tight_layout()
plt.show()