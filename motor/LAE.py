import numpy as np

def LAE(x, dt):
    K1 = 4200 # 3700 #
    K2 = 60 # 55 #與阻尼有關
    v = 0
    xe = x[0]
    ae = np.zeros((len(x)))
    for i in range(len(x)-1):
        ae[i+1] = K1*(x[i] - xe) - K2*v
        v += ae[i+1]*dt
        xe += v*dt
    return ae