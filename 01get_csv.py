__Author = "xq"


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.loadtxt('./Data20191209/SRC/TurnRight_Right.log', dtype="str")
L = []
R = []
for line in data:
    L_left = line[7]
    L_right = line[6]
    R_left = line[11]
    R_right = line[10]
    L_num = int("0b" + "{:08b}".format(int(L_left, 16)) + "{:08b}".format(int(L_right, 16)), 2)
    L.append(L_num)
    R_num = int("0b" + "{:08b}".format(int(R_left, 16)) + "{:08b}".format(int(R_right, 16)), 2)
    R.append(R_num)
    # print(num)

sub = pd.DataFrame({"L_num": L, "R_num": R})
# sub.to_csv("./Data20191209/TurnRight_Right.csv", index=False, header=True)
# x_l = len(L)
# x = np.linspace(0, x_l, x_l)
# plt.scatter(x, L)
# plt.show()

