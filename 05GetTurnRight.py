__Author__ = "xq"


import pandas as pd
import cv2

TURNLEFTPATN = "./Data20191117/DSTTurnLeft.csv"

TurnRightData = pd.DataFrame(columns=["TurnRight_img_path", "L_num", "R_num", "Steering"])
TurnLeftData = pd.read_csv(TURNLEFTPATN)

for i in range(TurnLeftData.shape[0]):
    data = TurnLeftData.iloc[i]
    L_img_path = "./Data20191117" + data[0]
    R_img_path = L_img_path[:19] + "TurnRight" + L_img_path[-9:]

    L_num = data[1]
    R_num = data[2]
    Steering = data[3]
    L_img = cv2.imread(L_img_path)
    R_img = cv2.flip(L_img, 1)

    cv2.imwrite(R_img_path, R_img)
    if not Steering == 0:
        Steering = -Steering
    TurnRightData.loc[i] = R_img_path[14:], R_num, L_num, Steering
TurnRightData.to_csv("./Data20191117/DSTTurnRight.csv", index=False, header=True)

