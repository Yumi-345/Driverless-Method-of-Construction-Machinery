__Author__ = "xq"


import pandas as pd
import cv2, time, threading
import numpy as np
import copy

avi_path = "./000.mp4"

video = cv2.VideoCapture(avi_path)
print(video.get(3))
print(video.get(4))
print(video.get(5))
i = 0
while True:
    _, frame = video.read()
    cv2.imshow("00", frame)
    print(frame.dtype)
    i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(i)
