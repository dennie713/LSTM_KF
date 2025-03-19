import numpy as np

def ImportData(path1, path2) :
    for i in range (len(path1)):
        #Motordata
        Motordata = []
        with open(path1[i], 'r')as f:
            # Motordata = f.read()
            lines = f.readlines()
            for line in lines:
                line = line.split()
                Motordata.append(line) #馬達資料
        #Mousedata
        Mousedata = []
        with open(path2[i], 'r')as f:
            # Mousedata = f.read()
            lines = f.readlines()
            for line in lines:
                line = line.split()
                Mousedata.append(line) #滑鼠資料

        ## Motordata, Mousedata    
        Motordata = np.array(Motordata)    
        Mousedata = np.array(Mousedata)
    return Motordata, Mousedata

def ImportData_2(path1, path2) :
    for i in range (len(path1)):
        #Motordata
        Motordata = []
        with open(path1, 'r')as f:
            # Motordata = f.read()
            lines = f.readlines()
            for line in lines:
                line = line.split()
                Motordata.append(line) #馬達資料
        #Mousedata
        Mousedata = []
        with open(path2, 'r')as f:
            # Mousedata = f.read()
            lines = f.readlines()
            for line in lines:
                line = line.split()
                Mousedata.append(line) #滑鼠資料

        ## Motordata, Mousedata    
        Motordata = np.array(Motordata)    
        Mousedata = np.array(Mousedata)
    return Motordata, Mousedata