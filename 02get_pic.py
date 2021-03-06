__Author = "xq"


import cv2

avi_path = "./Data20191209/SRC/TurnRight_Right.avi"

video = cv2.VideoCapture(avi_path)
i = 0
while True:
    _, frame = video.read()
    i += 1
    cv2.imshow("00", frame)
    cv2.imwrite("./Data20191209/IMG/TurnRight_Right/{:04d}.png".format(i), frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(i)