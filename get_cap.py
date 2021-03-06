__Author__ = "xq"


import cv2, time, threading

# 调用摄像头
videoCapture = cv2.VideoCapture(0)


# 设置帧率
fps = 30

# 获取窗口大小
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 调用VideoWrite（）函数
videoWrite = cv2.VideoWriter('TurnRight_Left.mp4', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, (640, 480))

# 先获取一帧，用来判断是否成功调用摄像头
success, frame = videoCapture.read()
# 通过设置帧数来设置时间,减一是因为上面已经获取过一帧了
# numFrameRemainling = fps * 50 - 1

# 通过循环保存帧
print(3)

time.sleep(1)
print(2)

time.sleep(1)
print(1)
while success:
    videoWrite.write(frame)
    # cv.imshow("00", frame)
    success, frame = videoCapture.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # numFrameRemainling -= 1

# 释放摄像头
videoCapture.release()

