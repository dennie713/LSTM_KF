def LSF14(xTemp):
    global AXIS, x, SamplingTime
    SamplingTime = 0.001
    y = []
    X = []
    AXIS = 1
    # xTemp = [[0.0] * 4 for _ in range(AXIS)]

    for i in range(len(xTemp)):
        if i>2 :
            temp = (0.3 * xTemp[i] + 0.1 * xTemp[i-1] - 0.1 * xTemp[i-2] - 0.3 * xTemp[i-3]) / SamplingTime
        else :
            temp = 0
        y.append(temp)
    return y

def LSF28(xTemp):
    global AXIS, x, SamplingTime
    SamplingTime = 0.001
    y = []
    X = []
    AXIS = 1
    # xTemp = [[0.0] * 4 for _ in range(AXIS)]

    for i in range(len(xTemp)):
        if i>6 :
            temp =(0.2083*xTemp[i-7]-0.0179*xTemp[i-6]
				  -0.1607*xTemp[i-5]-0.2202*xTemp[i-4]
			      -0.1964*xTemp[i-3]-0.0893*xTemp[i-2]
				  +0.1012*xTemp[i-1]+0.3750*xTemp[i])/SamplingTime
        else :
            temp = 0
        y.append(temp)
        # print(y)
    return y

def LSF38(xTemp):
    global AXIS, x, SamplingTime
    SamplingTime = 0.001
    y = []
    X = []
    AXIS = 1

    for i in range(len(xTemp)):
        if i>6 :
            temp =(-0.2778*xTemp[i-7]+0.3294*xTemp[i-6]
				  +0.3254*xTemp[i-5]-0.0119*xTemp[i-4]
			      -0.4048*xTemp[i-3]-0.5754*xTemp[i-2]
				  -0.2460*xTemp[i-1]+0.8611*xTemp[i])/SamplingTime
        else :
            temp = 0
        y.append(temp)
        # print(y)
    return y

def LSF28_Acc(xTemp):
    global AXIS, x, SamplingTime
    SamplingTime = 0.001
    y = []
    X = []
    AXIS = 1
    # xTemp = [[0.0] * 4 for _ in range(AXIS)]

    for i in range(len(xTemp)):
        if i>6 :
             temp = (0.0833*xTemp[i-7]+0.0119*xTemp[i-6]
                    -0.0357*xTemp[i-5]-0.0595*xTemp[i-4]
                    -0.0595*xTemp[i-3]-0.0357*xTemp[i-2]
	 		        +0.0119*xTemp[i-1]+0.0833*xTemp[i])/(SamplingTime*SamplingTime)
        else :
            temp = 0
        y.append(temp)
    return y
