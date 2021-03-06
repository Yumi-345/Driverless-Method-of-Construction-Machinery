__Author__ = "xq"


import pandas as pd
import os

data = pd.DataFrame(columns=["TurnRight_Right_img_path"])
i = 0
for img_name in os.listdir("./Data20191209/IMG/TurnRight_Right"):
    num = i * 1.6//1 + 1
    img_num = int(img_name.split(".")[0])
    if img_num == num:
        data.loc[i] = "/IMG/TurnRight_Right/" +  img_name
        i += 1

data_csv = pd.read_csv("./Data20191209/TurnRight_Right.csv")

# data_csv = data_csv.iloc[0:850]
data = pd.concat((data, data_csv), axis=1, ignore_index=False)

# steering = (data.L_num - data.R_num)/65536
steering = (data.L_num - data.R_num)/65536 - 0.1
steering = pd.DataFrame(steering, columns=["Steering"])
data = pd.concat((data, steering), axis=1)

data.to_csv("./Data20191209/DSTTurnRight_Right.csv", index=False, header=True)
print(data.info())


