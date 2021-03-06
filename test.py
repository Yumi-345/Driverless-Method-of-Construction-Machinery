__Author__ = "xq"


from keras.models import load_model
import numpy as np
import cv2, os

model = load_model("./Data20191117/Model/Model.h5")

cap = cv2.VideoCapture("./Data20191117/SRC/Right.avi")
# cap = cv2.VideoCapture(1)

def getCan(num):
    string = "{:016b}".format(int(num * 65535 // 1))
    L = string[8:]
    R = string[:8]
    return str(hex(int(L, 2)))[2:].upper() + str(hex(int(R, 2)))[2:].upper()

def getLR(steering):
    if steering > 0:
        R = 0.5 - steering
        L = 0.5
    else:
        R = 0.5
        L = 0.5 + steering
    L = getCan(L)
    R = getCan(R)
    return L, R

while True:
    _, frame = cap.read()
    cv2.imshow("00", frame)
    frame = (frame / 255.0).astype(np.float32)
    frame = np.expand_dims(frame, axis=0)
    STEERING = model.predict(frame, batch_size=1)
    print(STEERING)

    L, R = getLR(STEERING)
    CAN = L + "0000" + R + "0000"
    print(CAN)
    # os.system("cansend can0 0x18FF33F0#" + CAN)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

