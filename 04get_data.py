__Author__ = "xq"


import pandas as pd

DSTTurnLeft = pd.read_csv("./Data20191209/DSTTurnLeft.csv", header=None, skiprows=1).iloc[80:-80,:]
DSTTurnLeft_Left = pd.read_csv("./Data20191209/DSTTurnLeft_Left.csv", header=None, skiprows=1).iloc[80:-80,:]
DSTTurnLeft_Right = pd.read_csv("./Data20191209/DSTTurnLeft_Right.csv", header=None, skiprows=1).iloc[80:-80,:]

DSTTurnRight = pd.read_csv("./Data20191209/DSTTurnRight.csv", header=None, skiprows=1).iloc[80:-80,:]
DSTTurnRight_Left = pd.read_csv("./Data20191209/DSTTurnRight_Left.csv", header=None, skiprows=1).iloc[80:-80,:]
DSTTurnRight_Right = pd.read_csv("./Data20191209/DSTTurnRight_Right.csv", header=None, skiprows=1).iloc[80:-80,:]

# DSTTurnLeft = pd.read_csv("./Data20191209/DSTTurnLeft.csv", header=None, skiprows=1).iloc[300:1000,:]
# DSTTurnRight = pd.read_csv("./Data20191209/DSTTurnRight.csv", header=None, skiprows=1).iloc[300:1000,:]


data = pd.concat((DSTTurnLeft, DSTTurnLeft_Left, DSTTurnLeft_Right, DSTTurnRight, DSTTurnRight_Left, DSTTurnRight_Right),
                 axis=0, ignore_index=True)
data.columns = ["IMGPath", "L_num", "R_num", "Steering"]
data.to_csv("./Data20191209/Data.csv", index=False, header=True)

